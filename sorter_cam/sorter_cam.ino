/*
 * Resistor sorter - CAMERA side (AI-Thinker ESP32-CAM)
 *
 * Captures QVGA (320x240) JPEG frames as fast as possible and streams them
 * over UART to the ESP32-S3, which decodes them and runs the classifier
 * (sorter_s3.ino). No processing happens here, so the camera never waits on
 * the model: the UART link is the only limit (~20-25 fps at 2 Mbaud).
 *
 * Wiring (3 wires):
 *   ESP32-CAM U0T (GPIO1)  ->  ESP32-S3 GPIO18  (S3 RX)
 *   ESP32-CAM GND          ->  ESP32-S3 GND     (required!)
 *   ESP32-CAM 5V           <-  5V supply (the camera needs a solid 5V/500mA)
 *   (S3 TX -> CAM U0R is not needed; leave it unconnected so it can't
 *    interfere when you flash the ESP32-CAM.)
 *
 * Arduino IDE settings (Tools menu):
 *   Board:            "AI Thinker ESP32-CAM"
 *   PSRAM:            Enabled
 *   Core Debug Level: "None"   <-- debug text would corrupt the stream
 *
 * Packet per frame (same format as esp32cam_uart_stream.ino, so uart_viewer.py
 * can still show this stream on a PC for capturing more training images):
 *   A5 5A C3 3C | seq u16 | len u32 | JPEG bytes | sum16 u16   (little-endian)
 */
#include "esp_camera.h"
#include "esp_log.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ===================== USER SETTINGS =====================
#define LINK_BAUD     2000000  // must match sorter_s3.ino. 4000000 works with short (<15 cm) wires
#define USE_UART0     1        // 1 = U0T/GPIO1 pin (labeled on the board)
                               // 0 = GPIO14 (frees UART0 for debug prints)
#define UART2_TX_PIN  14
#define JPEG_QUALITY  12       // must stay 12: the model was trained on this compression level
#define FLASH_LED     0        // 0-255 brightness of the white LED (GPIO4). Use the same lighting
                               // as when the training images were taken.
// =========================================================

#if USE_UART0
  #define LINK Serial
#else
  #define LINK Serial2
#endif

// AI-Thinker ESP32-CAM camera pins
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22
#define FLASH_GPIO_NUM  4

static bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;     // hardware JPEG: no CPU cost, ~8 KB per frame
  c.frame_size   = FRAMESIZE_QVGA;     // 320x240, what the model expects
  c.jpeg_quality = JPEG_QUALITY;
  c.fb_count     = 2;                  // capture the next frame while this one is sent
  c.fb_location  = CAMERA_FB_IN_PSRAM;
  c.grab_mode    = CAMERA_GRAB_LATEST;  // always send the newest frame (lowest latency)
  return esp_camera_init(&c) == ESP_OK;
}

static void sendFrame(const uint8_t *d, size_t n) {
  static uint16_t seq = 0;
  uint8_t h[10] = {0xA5, 0x5A, 0xC3, 0x3C, (uint8_t)seq, (uint8_t)(seq >> 8),
                   (uint8_t)n, (uint8_t)(n >> 8), (uint8_t)(n >> 16), (uint8_t)(n >> 24)};
  seq++;
  uint16_t s = 0;
  for (size_t i = 0; i < n; i++) s += d[i];
  uint8_t t[2] = {(uint8_t)s, (uint8_t)(s >> 8)};
  LINK.write(h, sizeof(h));
  LINK.write(d, n);          // blocks while the TX buffer is full: natural flow control
  LINK.write(t, sizeof(t));
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);   // camera start-up current often trips brown-out
  esp_log_level_set("*", ESP_LOG_NONE);        // nothing but frame packets on the wire

  LINK.setTxBufferSize(16384);
#if USE_UART0
  LINK.begin(LINK_BAUD);
  LINK.setDebugOutput(false);
#else
  LINK.begin(LINK_BAUD, SERIAL_8N1, -1, UART2_TX_PIN);
  Serial.begin(115200);
  Serial.println("sorter_cam: streaming on GPIO14");
#endif

  if (!initCamera()) {
    // camera failure: blink the flash LED once a second forever
    pinMode(FLASH_GPIO_NUM, OUTPUT);
    for (;;) { digitalWrite(FLASH_GPIO_NUM, HIGH); delay(30); digitalWrite(FLASH_GPIO_NUM, LOW); delay(970); }
  }
  pinMode(FLASH_GPIO_NUM, OUTPUT);
  digitalWrite(FLASH_GPIO_NUM, LOW);
  if (FLASH_LED > 0) {
    // channel 7 (timer 3): channel 0 / timer 0 drive the camera clock
    ledcAttachChannel(FLASH_GPIO_NUM, 5000, 8, 7);
    ledcWrite(FLASH_GPIO_NUM, FLASH_LED);
  }

  for (int i = 0; i < 5; i++) {   // let auto exposure settle
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { delay(2); return; }
  sendFrame(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}
