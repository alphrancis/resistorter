/*
 * ESP32-CAM (AI-Thinker) -> JPEG video stream over UART
 *
 * Each frame is sent as a binary packet:
 *
 *   [0..3]   MAGIC      A5 5A C3 3C
 *   [4..5]   seq        uint16, little-endian (frame counter)
 *   [6..9]   length     uint32, little-endian (JPEG size in bytes)
 *   [10..]   JPEG data  (length bytes)
 *   [+2]     checksum   uint16, little-endian (sum of all JPEG bytes, mod 65536)
 *
 * Arduino IDE settings:
 *   Board:            "AI Thinker ESP32-CAM"
 *   Core Debug Level: "None"   <-- important, debug text would corrupt the stream
 *   PSRAM:            Enabled
 *
 * Throughput: 921600 baud ~ 90 KB/s. A QVGA JPEG is ~5-10 KB, so expect ~8-15 fps.
 * Raise STREAM_BAUD (e.g. 2000000) if your USB-serial adapter supports it.
 */

#include "esp_camera.h"
#include "esp_log.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ===================== USER SETTINGS =====================
// 1 = UART0 (U0TXD = GPIO1, U0RXD = GPIO3) - the same pins you flash with.
// 0 = UART2 on the pins below (keeps UART0 free for debugging).
#define USE_UART0      1
#define UART2_TX_PIN   14   // safe, not a boot strapping pin
#define UART2_RX_PIN   15   // only used for input

#define STREAM_BAUD    921600

#define FRAME_SIZE     FRAMESIZE_SVGA  // QQVGA(160x120) QVGA(320x240) VGA(640x480) SVGA(800x600) UXGA(1600x1200)
#define JPEG_QUALITY   10             // 10-63, lower = better quality / bigger frames
#define MAX_FPS        0               // 0 = as fast as the UART allows
// =========================================================

#if USE_UART0
  #define LINK Serial
#else
  #define LINK Serial2
#endif

// AI-Thinker ESP32-CAM camera pins
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

static const uint8_t MAGIC[4] = {0xA5, 0x5A, 0xC3, 0x3C};
static uint16_t frameSeq = 0;

static bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;   // older cores: pin_sscb_sda
  config.pin_sccb_scl = SIOC_GPIO_NUM;   // older cores: pin_sscb_scl
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;

  if (psramFound()) {
    config.fb_count    = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode   = CAMERA_GRAB_LATEST;  // always send the newest frame
  } else {
    config.fb_count    = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.grab_mode   = CAMERA_GRAB_WHEN_EMPTY;
    if (FRAME_SIZE > FRAMESIZE_QVGA) config.frame_size = FRAMESIZE_QVGA;
  }

  return esp_camera_init(&config) == ESP_OK;
}

static inline void writeU16(uint8_t *p, uint16_t v) { p[0] = v & 0xFF; p[1] = v >> 8; }
static inline void writeU32(uint8_t *p, uint32_t v) {
  p[0] = v & 0xFF; p[1] = (v >> 8) & 0xFF; p[2] = (v >> 16) & 0xFF; p[3] = v >> 24;
}

static void sendFrame(const uint8_t *data, size_t len) {
  uint8_t header[10];
  memcpy(header, MAGIC, 4);
  writeU16(header + 4, frameSeq++);
  writeU32(header + 6, (uint32_t)len);

  uint16_t sum = 0;
  for (size_t i = 0; i < len; i++) sum += data[i];
  uint8_t trailer[2];
  writeU16(trailer, sum);

  LINK.write(header, sizeof(header));
  LINK.write(data, len);            // blocks until queued; that's our flow control
  LINK.write(trailer, sizeof(trailer));
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);   // ESP32-CAM often browns out at camera start

  // Silence every log source so only frame packets go out on the wire
  esp_log_level_set("*", ESP_LOG_NONE);

  LINK.setTxBufferSize(8192);
#if USE_UART0
  LINK.begin(STREAM_BAUD);
  LINK.setDebugOutput(false);
#else
  LINK.begin(STREAM_BAUD, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);
  Serial.begin(115200);   // UART0 free for your own debug prints
  Serial.println("UART2 stream starting");
#endif

  if (!initCamera()) {
    // Blink the flash LED (GPIO4) briefly forever to signal camera failure
    pinMode(4, OUTPUT);
    while (true) { digitalWrite(4, HIGH); delay(30); digitalWrite(4, LOW); delay(970); }
  }

  // Throw away the first few frames (auto-exposure settling)
  for (int i = 0; i < 3; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
}

void loop() {

  static uint32_t lastFrameMs = 0;
#if MAX_FPS > 0
  if (millis() - lastFrameMs < (1000 / MAX_FPS)) return;
#endif
  lastFrameMs = millis();

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { delay(5); return; }

  sendFrame(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}
