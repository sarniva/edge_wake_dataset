/*
 * edge_wake_dataset: field recorder for KWS dataset collection.
 *
 * Records 30 s (or 5 s quick-check) takes of 16 kHz mono PCM from INMP441
 * and hex-dumps them over serial. Works over NATIVE USB (USB-Serial/JTAG):
 * open the port in any serial terminal (laptop or Android OTG app), type
 * a command, save the log, convert with scripts/capture_takes.py.
 *
 * Wiring (INMP441 breakout -> ESP32-S3 DevKit, same as edge_wake):
 *   VDD -> 3V3 (NEVER 5V) | GND -> GND | L/R -> GND (LEFT slot)
 *   SD -> GPIO10 | WS -> GPIO11 | SCK -> GPIO12
 *
 * Use:
 *   BOOT tap ............... 5 s quick take (level check)
 *   BOOT hold 1.5 s ........ 30 s take (dataset material)
 *   power-on / USB plug .... auto 30 s take (AUTO_TAKE_ON_BOOT, field use:
 *                            plug into phone, save log, done - no interaction)
 *   serial 'r' / 'q' ....... same as hold / tap (works where host USB writes
 *                            function; some hosts stall on CDC-ACM OUT - the
 *                            button path never needs writes)
 *   VU lines every 500 ms show live level; VAD flips on speech.
 *   Takes are numbered (TAKE_START n=N) so one saved log holds many takes.
 *
 * Conversion: 24-bit-in-32-bit left slot >> 12 with saturation (~+24 dB,
 * compensates the mic's -26 dBFS sensitivity). Clipping on taps is normal;
 * constant full-scale on speech means back off or rebuild with shift 13.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"

static const char *TAG = "dataset";

#define PIN_I2S_BCLK   GPIO_NUM_12
#define PIN_I2S_WS     GPIO_NUM_11
#define PIN_I2S_DIN    GPIO_NUM_10
#define PIN_BUTTON     GPIO_NUM_0

#define SAMPLE_RATE     16000
#define GAIN_SHIFT      12
#define HOP_SAMPLES     480          // 30 ms
#define TAKE30_SAMPLES  (16000 * 30) // 480000 samples, 960 KB (PSRAM)
#define TAKE5_SAMPLES   (16000 * 5)  // 80000 samples (fits internal, lives in same buf)

// Field mode: one automatic 30 s take shortly after boot. This is what makes
// phone collection zero-interaction (plug in -> take -> save log) and gives
// hosts with broken CDC-ACM writes a trigger that needs no writes at all.
#define AUTO_TAKE_ON_BOOT 1
#define AUTO_TAKE_DELAY_MS 2000

// VAD calibrated 2026-09-04, traffic-background room (see edge_wake Phase 1)
#define VAD_ENTER_RMS     6000
#define VAD_EXIT_RMS      3000
#define VAD_HANGOVER_HOPS 10

static i2s_chan_handle_t rx_chan = NULL;
static int16_t *take_buf = NULL;
static bool have_serial_cmd = false;
static int take_no = 0;

typedef enum { VAD_SILENCE = 0, VAD_SPEECH = 1 } vad_state_t;
static vad_state_t vad_state = VAD_SILENCE;
static int vad_below_cnt = 0;

static inline int16_t conv_s32_to_s16(int32_t s32)
{
    int32_t v = s32 >> GAIN_SHIFT;
    if (v > 32767)  v = 32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

static void i2s_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan_cfg.dma_frame_num = HOP_SAMPLES;
    chan_cfg.dma_desc_num  = 6;
    chan_cfg.auto_clear    = true;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = PIN_I2S_BCLK,
            .ws   = PIN_I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = PIN_I2S_DIN,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
    ESP_LOGI(TAG, "I2S RX ready: %d Hz, BCLK=%d WS=%d DIN=%d",
             SAMPLE_RATE, PIN_I2S_BCLK, PIN_I2S_WS, PIN_I2S_DIN);
}

static int read_mono_block(int16_t *out, int n_frames)
{
    static int32_t *raw = NULL;
    static size_t raw_cap_frames = 0;
    if (raw_cap_frames < (size_t)n_frames) {
        free(raw);
        raw = malloc(n_frames * 2 * sizeof(int32_t));
        if (!raw) { ESP_LOGE(TAG, "no mem for raw block"); return 0; }
        raw_cap_frames = n_frames;
    }
    size_t bytes_want = n_frames * 2 * sizeof(int32_t);
    size_t bytes_read = 0;
    esp_err_t r = i2s_channel_read(rx_chan, (uint8_t *)raw, bytes_want,
                                   &bytes_read, pdMS_TO_TICKS(1000));
    if (r != ESP_OK || bytes_read == 0) return 0;
    int got_frames = bytes_read / (2 * sizeof(int32_t));
    for (int i = 0; i < got_frames; i++) {
        out[i] = conv_s32_to_s16(raw[2 * i]);   // LEFT slot (L/R pin = GND)
    }
    return got_frames;
}

static int hop_rms(const int16_t *pcm, int n)
{
    long long sumsq = 0;
    for (int i = 0; i < n; i++) sumsq += (long long)pcm[i] * pcm[i];
    return (int)sqrt((double)sumsq / n);
}

static bool vad_update(int rms)
{
    if (vad_state == VAD_SILENCE) {
        if (rms >= VAD_ENTER_RMS) { vad_state = VAD_SPEECH; vad_below_cnt = 0; return true; }
    } else if (rms < VAD_EXIT_RMS) {
        if (++vad_below_cnt >= VAD_HANGOVER_HOPS) {
            vad_state = VAD_SILENCE; vad_below_cnt = 0; return true;
        }
    } else {
        vad_below_cnt = 0;
    }
    return false;
}

static uint32_t pcm_crc32(const int16_t *pcm, int n)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (int i = 0; i < n; i++) {
        crc ^= (uint16_t)pcm[i];
        for (int b = 0; b < 16; b++) crc = (crc & 1) ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
    }
    return ~crc;
}

// Forward declarations.
static void do_take(int seconds, const char *why);

static void poll_serial_cmd(void)
{
    if (!have_serial_cmd) return;
    char c;
    if (read(STDIN_FILENO, &c, 1) != 1) return;
    if (c == 'r') {
        ESP_LOGI(TAG, "serial cmd 'r' -> 30 s take");
        do_take(30, "serial");
    } else if (c == 'q') {
        ESP_LOGI(TAG, "serial cmd 'q' -> 5 s quick take");
        do_take(5, "serial");
    }
}

static void do_take(int seconds, const char *why)
{
    int want = SAMPLE_RATE * seconds;
    take_no++;
    ESP_LOGI(TAG, "=== take %d (%s): recording %d s ===", take_no, why, seconds);
    static int16_t trash[480];
    for (int i = 0; i < 4; i++) read_mono_block(trash, 480);  // flush pipeline
    printf("TAKE_START n=%d seconds=%d\n", take_no, seconds);
    printf("REC_START seconds=%d\n", seconds);

    int filled = 0;
    int64_t t0 = esp_timer_get_time();
    while (filled < want) {
        int chunk = want - filled > 960 ? 960 : want - filled;
        int got = read_mono_block(take_buf + filled, chunk);
        if (got <= 0) { ESP_LOGW(TAG, "i2s timeout at %d/%d", filled, want); continue; }
        filled += got;
        if ((filled / 960) % 16 == 0) esp_task_wdt_reset();
    }
    int64_t t1 = esp_timer_get_time();
    printf("REC_END samples=%d\n", filled);

    long long sumsq = 0;
    int peak = 0;
    for (int i = 0; i < filled; i++) {
        sumsq += (long long)take_buf[i] * take_buf[i];
        int a = take_buf[i] >= 0 ? take_buf[i] : -take_buf[i];
        if (a > peak) peak = a;
    }
    double rms = sqrt((double)sumsq / filled);
    ESP_LOGI(TAG, "take %d: %d samples in %.2f s, rms=%.0f peak=%d %.1f dBFS",
             take_no, filled, (t1 - t0) / 1000000.0, rms, peak,
             (rms < 0.5) ? -96.0 : 20.0 * log10(rms / 32768.0));

    printf("AUD_DUMP_START samples=%d sr=%d bits=16 ch=1 shift=%d take=%d\n",
           filled, SAMPLE_RATE, GAIN_SHIFT, take_no);
    for (int i = 0; i < filled; i += 256) {
        int m = filled - i < 256 ? filled - i : 256;
        printf("AUD_DATA:");
        for (int j = 0; j < m; j++) printf("%04x", (uint16_t)take_buf[i + j]);
        printf("\n");
        // The dump is CPU-bound printf for ~2 MB: yield regularly or the
        // task watchdog fires its warning text MID-DUMP (corrupts a line).
        if ((i / 256) % 16 == 15) { esp_task_wdt_reset(); vTaskDelay(pdMS_TO_TICKS(5)); }
    }
    printf("AUD_DUMP_END\n");
    printf("AUD_CRC %08lx take=%d\n", (unsigned long)pcm_crc32(take_buf, filled), take_no);
    printf("TAKE_END n=%d samples=%d\n", take_no, filled);
    fflush(stdout);
    ESP_LOGI(TAG, "take %d dumped. 'r'=30 s take, 'q'=5 s take", take_no);
}

void app_main(void)
{
    ESP_LOGI(TAG, "edge_wake_dataset boot. heap internal=%u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    // Subscribe the main task to the task watchdog: the 30 s capture and
    // the ~2 MB dump loops must feed it explicitly (printf over USB-CDC can
    // spinwait, starving IDLE until the WDT text lands mid-dump).
    esp_err_t wdt = esp_task_wdt_add(NULL);
    ESP_LOGI(TAG, "task-wdt subscribe: %s", esp_err_to_name(wdt));

    gpio_set_direction(PIN_BUTTON, GPIO_MODE_INPUT);
    gpio_set_pull_mode(PIN_BUTTON, GPIO_PULLUP_ONLY);

    int fl = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (fl >= 0 && fcntl(STDIN_FILENO, F_SETFL, fl | O_NONBLOCK) == 0) {
        have_serial_cmd = true;
    }

    i2s_init();

    take_buf = heap_caps_malloc(TAKE30_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!take_buf) take_buf = heap_caps_malloc(TAKE30_SAMPLES * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!take_buf) take_buf = malloc(TAKE30_SAMPLES * sizeof(int16_t));
    if (!take_buf) { ESP_LOGE(TAG, "cannot allocate take buffer"); return; }
    ESP_LOGI(TAG, "take buffer %u KB @ %p", (unsigned)(TAKE30_SAMPLES * 2 / 1024), take_buf);

    printf("\n==== DATASET RECORDER READY ====\n"
           "'r' = 30 s take | 'q' = 5 s quick take | BOOT tap = 5 s | BOOT hold 1.5 s = 30 s\n"
           "Say 5-8x JAGO GURU per 30 s take, 3-4 s apart. Save this log, convert with scripts/capture_takes.py\n"
           "===============================\n");
    fflush(stdout);

#if AUTO_TAKE_ON_BOOT
    vTaskDelay(pdMS_TO_TICKS(AUTO_TAKE_DELAY_MS));  // let levels/host settle
    do_take(30, "boot");
#endif

    static int16_t hop[HOP_SAMPLES];
    int btn_prev = 1;
    int64_t t_boot = esp_timer_get_time();
    int64_t last_vu = t_boot;

    ESP_LOGI(TAG, "serial commands %s", have_serial_cmd ? "ON ('r'/'q')" : "OFF (buttons only)");
    while (1) {
        poll_serial_cmd();
        int got = read_mono_block(hop, HOP_SAMPLES);
        if (got <= 0) { vTaskDelay(pdMS_TO_TICKS(5)); continue; }

        int rms = hop_rms(hop, got);
        // Gate VAD until DMA settles (~10 hops of startup garbage).
        static uint32_t hops = 0;
        hops++;
        if (hops > 10 && vad_update(rms)) {
            ESP_LOGI(TAG, "VAD: -> %s (rms=%d, t=%.2fs)",
                     vad_state == VAD_SPEECH ? "SPEECH" : "SILENCE",
                     rms, (esp_timer_get_time() - t_boot) / 1000000.0);
        }

        int64_t now = esp_timer_get_time();
        if (now - last_vu > 500000) {
            last_vu = now;
            long long sumsq = 0; int peak = 0;
            for (int i = 0; i < got; i++) {
                sumsq += (long long)hop[i] * hop[i];
                int a = hop[i] >= 0 ? hop[i] : -hop[i];
                if (a > peak) peak = a;
            }
            double r = sqrt((double)sumsq / got);
            ESP_LOGI(TAG, "VU rms=%.0f peak=%d %.1f dBFS vad=%s",
                     r, peak, (r < 0.5) ? -96.0 : 20.0 * log10(r / 32768.0),
                     vad_state == VAD_SPEECH ? "SPEECH" : "silence");
        }

        int btn = gpio_get_level(PIN_BUTTON);
        if (btn_prev == 1 && btn == 0) {
            int64_t t_press = esp_timer_get_time();
            vTaskDelay(pdMS_TO_TICKS(50));
            if (gpio_get_level(PIN_BUTTON) != 0) { btn_prev = 1; continue; }
            bool hold = false;
            while (gpio_get_level(PIN_BUTTON) == 0) {
                if (esp_timer_get_time() - t_press > 1500000) { hold = true; break; }
                vTaskDelay(pdMS_TO_TICKS(20));
            }
            if (hold) do_take(30, "button-hold");
            else do_take(5, "button");
            while (gpio_get_level(PIN_BUTTON) == 0) vTaskDelay(pdMS_TO_TICKS(20));
            btn_prev = 1;
            last_vu = esp_timer_get_time();
            continue;
        }
        btn_prev = btn;
    }
}
