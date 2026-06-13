// ============================================================================
//  VoiceLink — Wearable Capture Device  (ESP32-S3)
// ----------------------------------------------------------------------------
//  Captures one VGA JPEG frame every FRAME_INTERVAL_MS and continuously
//  captures 16 kHz mono audio from an SPH0645 I2S mic. Both streams go out over
//  a SINGLE WebSocket while the device runs as its own Wi-Fi access point
//  (SoftAP) — so it works with no router (the portable, body-worn case).
//
//  This device does NO processing. All intelligence lives on the Mac.
//  (See CLAUDE.md — "three devices, one seam".)
//
//  Binary WebSocket message format — 1-byte type tag followed by payload:
//      0x01  + JPEG bytes   -> one camera frame
//      0x02  + PCM bytes    -> a chunk of 16-bit little-endian mono audio
//
//  Concurrency: the I2S audio read+send runs in its own high-priority FreeRTOS
//  task pinned to CORE 0. The camera capture, WebSocket servicing, and button
//  run in the Arduino loop() on CORE 1. This keeps audio from stuttering when a
//  (relatively slow) JPEG frame is grabbed and sent.
//
//  ---- Build setup (Arduino IDE) --------------------------------------------
//    Board:  "ESP32S3 Dev Module"  (Freenove ESP32-S3-WROOM CAM)
//    PSRAM:  "OPI PSRAM"            <-- REQUIRED, or camera init fails
//    Library: "WebSockets" by Markus Sattler  (a.k.a. arduinoWebSockets)
//
//  ---- Implementation notes (assumptions stated, per CLAUDE.md) -------------
//    * Uses the legacy `driver/i2s.h` API: it compiles on both Arduino-ESP32
//      2.x and 3.x and is the canonical SPH0645 path. On a future core where it
//      is removed, port the i2s_* calls to `driver/i2s_std.h` — the SPH0645
//      sample math below is unaffected (we process raw 32-bit words ourselves).
//    * The recording LED defaults to the on-board WS2812 (no wiring, very
//      visible) via the core builtin rgbLedWrite(). Set LED_TYPE = LED_SIMPLE
//      for a plain LED on a digital pin.
// ============================================================================

#include "esp_camera.h"
#include <WiFi.h>
#include <WebSocketsServer.h>
#include "driver/i2s.h"

// ============================================================================
//  CONFIG  — every pin, credential, and tunable lives here. Edit this block.
// ============================================================================

// --- Wi-Fi SoftAP (the device creates its own network; no router needed) ----
static const char*    AP_SSID     = "VoiceLink-Capture";
static const char*    AP_PASSWORD = "voicelink";   // >= 8 chars, or "" for open
static const uint8_t  AP_CHANNEL  = 6;
static const uint16_t WS_PORT     = 81;            // ws://192.168.4.1:81/

// --- Camera frame cadence ----------------------------------------------------
static const uint32_t   FRAME_INTERVAL_MS = 2000;  // one VGA JPEG every 2 s
static const framesize_t FRAME_SIZE       = FRAMESIZE_VGA;  // 640x480
static const int        JPEG_QUALITY      = 12;    // 10 (best) .. 63 (worst)

// --- Microphone: SPH0645 I2S MEMS, SEL tied to GND -> LEFT channel ----------
static const i2s_port_t I2S_PORT     = I2S_NUM_0;
static const int        MIC_BCLK_PIN = 1;
static const int        MIC_WS_PIN   = 2;
static const int        MIC_DIN_PIN  = 42;
static const uint32_t   SAMPLE_RATE  = 16000;      // Hz, mono
static const int        AUDIO_CHUNK  = 512;        // samples per WS message
// SPH0645 quirk: each 18-bit sample arrives left-justified in a 32-bit word.
// We read 32-bit words and arithmetic-shift right by this many bits. Increase
// to lower the audio level if loud sounds clip.
static const int        MIC_SHIFT_BITS = 14;

// --- Recording-indicator LED (ON whenever capture is active) ----------------
//  On a body-worn device this LED is the visible signal to people nearby that
//  capture is on. Keep it obvious.
#define LED_RGB    1   // on-board addressable WS2812 (no library; core builtin)
#define LED_SIMPLE 2   // a plain LED on a digital pin
static const int  LED_TYPE        = LED_RGB;
static const int  LED_PIN         = 48;    // WS2812 on most S3 devkits; or your LED pin
static const bool LED_ACTIVE_HIGH = true;  // LED_SIMPLE only
static const uint8_t LED_RED_BRIGHTNESS = 64;  // 0..255 (WS2812 red = "recording")

// --- Capture toggle button (wired to GND; uses the internal pull-up) --------
static const int      BUTTON_PIN  = 14;
static const uint32_t DEBOUNCE_MS = 50;

// --- Camera pin map: CAMERA_MODEL_ESP32S3_EYE -------------------------------
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   15
#define SIOD_GPIO_NUM    4
#define SIOC_GPIO_NUM    5
#define Y9_GPIO_NUM     16
#define Y8_GPIO_NUM     17
#define Y7_GPIO_NUM     18
#define Y6_GPIO_NUM     12
#define Y5_GPIO_NUM     10
#define Y4_GPIO_NUM      8
#define Y3_GPIO_NUM      9
#define Y2_GPIO_NUM     11
#define VSYNC_GPIO_NUM   6
#define HREF_GPIO_NUM    7
#define PCLK_GPIO_NUM   13

// ============================================================================
//  END CONFIG
// ============================================================================

static const uint8_t MSG_FRAME = 0x01;
static const uint8_t MSG_AUDIO = 0x02;

// Reusable PSRAM buffer for "tag byte + JPEG" so we send one tagged message
// without re-allocating per frame.
static const size_t FRAME_SEND_BUF_MAX = 120 * 1024;

WebSocketsServer  webSocket = WebSocketsServer(WS_PORT);
SemaphoreHandle_t wsMutex   = nullptr;   // arduinoWebSockets is not thread-safe
volatile bool     capturing = true;      // toggled by the button
bool              cameraReady = false;
uint8_t*          frameSendBuf = nullptr;
uint32_t          lastFrameMs  = 0;

// ----------------------------------------------------------------------------
//  Recording-indicator LED
// ----------------------------------------------------------------------------
static void ledInit() {
  if (LED_TYPE == LED_SIMPLE) {
    pinMode(LED_PIN, OUTPUT);
  }
  // LED_RGB uses rgbLedWrite() (Arduino-ESP32 builtin) — no init required.
}

static void ledSet(bool on) {
  if (LED_TYPE == LED_RGB) {
    rgbLedWrite(LED_PIN, on ? LED_RED_BRIGHTNESS : 0, 0, 0);  // red = recording
  } else {
    digitalWrite(LED_PIN, (on == LED_ACTIVE_HIGH) ? HIGH : LOW);
  }
}

// ----------------------------------------------------------------------------
//  Camera
// ----------------------------------------------------------------------------
static bool cameraInit() {
  camera_config_t cfg = {};
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0 = Y2_GPIO_NUM;
  cfg.pin_d1 = Y3_GPIO_NUM;
  cfg.pin_d2 = Y4_GPIO_NUM;
  cfg.pin_d3 = Y5_GPIO_NUM;
  cfg.pin_d4 = Y6_GPIO_NUM;
  cfg.pin_d5 = Y7_GPIO_NUM;
  cfg.pin_d6 = Y8_GPIO_NUM;
  cfg.pin_d7 = Y9_GPIO_NUM;
  cfg.pin_xclk     = XCLK_GPIO_NUM;
  cfg.pin_pclk     = PCLK_GPIO_NUM;
  cfg.pin_vsync    = VSYNC_GPIO_NUM;
  cfg.pin_href     = HREF_GPIO_NUM;
  cfg.pin_sccb_sda = SIOD_GPIO_NUM;
  cfg.pin_sccb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn     = PWDN_GPIO_NUM;
  cfg.pin_reset    = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 20000000;
  cfg.pixel_format = PIXFORMAT_JPEG;
  cfg.frame_size   = FRAME_SIZE;
  cfg.jpeg_quality = JPEG_QUALITY;
  cfg.fb_count     = 2;
  cfg.fb_location  = CAMERA_FB_IN_PSRAM;
  cfg.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("[CAM] init FAILED: 0x%x (%s)\n", err, esp_err_to_name(err));
    Serial.println("[CAM] >>> Set Tools -> PSRAM to \"OPI PSRAM\" and reseat the camera ribbon.");
    return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
//  Microphone (I2S RX)
// ----------------------------------------------------------------------------
static bool micInit() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,   // SPH0645 packs 18 bits / 32-bit slot
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,    // SEL->GND => left channel
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  if (i2s_driver_install(I2S_PORT, &cfg, 0, NULL) != ESP_OK) return false;

  i2s_pin_config_t pins = {
    .mck_io_num   = I2S_PIN_NO_CHANGE,
    .bck_io_num   = MIC_BCLK_PIN,
    .ws_io_num    = MIC_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = MIC_DIN_PIN
  };
  if (i2s_set_pin(I2S_PORT, &pins) != ESP_OK) return false;

  i2s_zero_dma_buffer(I2S_PORT);
  return true;
}

// High-priority audio task, pinned to core 0. Reads I2S, converts SPH0645 raw
// words to 16-bit PCM, and streams them. Never blocks on the camera.
static void audioTask(void* arg) {
  static int32_t raw[AUDIO_CHUNK];
  static uint8_t sendBuf[1 + AUDIO_CHUNK * 2];     // tag + int16 PCM
  int16_t* pcm = (int16_t*)(sendBuf + 1);
  sendBuf[0] = MSG_AUDIO;

  static int32_t dcOffset = 0;                     // slow DC tracker (high-pass)
  size_t bytesRead = 0;

  for (;;) {
    // Always drain the DMA buffers (keeps timing clean across pause/resume),
    // but only convert + send while capturing.
    esp_err_t r = i2s_read(I2S_PORT, raw, sizeof(raw), &bytesRead, pdMS_TO_TICKS(100));
    if (r != ESP_OK || bytesRead == 0) continue;
    if (!capturing) continue;

    int n = bytesRead / sizeof(int32_t);
    for (int i = 0; i < n; i++) {
      int32_t s = raw[i] >> MIC_SHIFT_BITS;        // arithmetic shift (signed)
      dcOffset += (s - dcOffset) >> 6;             // remove DC offset
      int32_t c = s - dcOffset;
      if (c > 32767)  c = 32767;
      if (c < -32768) c = -32768;
      pcm[i] = (int16_t)c;
    }

    if (xSemaphoreTake(wsMutex, portMAX_DELAY) == pdTRUE) {
      webSocket.broadcastBIN(sendBuf, 1 + n * 2);
      xSemaphoreGive(wsMutex);
    }
  }
}

// ----------------------------------------------------------------------------
//  WebSocket events (logging only)
// ----------------------------------------------------------------------------
static void onWsEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t len) {
  switch (type) {
    case WStype_CONNECTED: {
      IPAddress ip = webSocket.remoteIP(num);
      Serial.printf("[WS] client #%u connected from %s\n", num, ip.toString().c_str());
      break;
    }
    case WStype_DISCONNECTED:
      Serial.printf("[WS] client #%u disconnected\n", num);
      break;
    default:
      break;
  }
}

// ----------------------------------------------------------------------------
//  Button: debounced, toggles capture on each press (active-low)
// ----------------------------------------------------------------------------
static void serviceButton() {
  static bool     lastReading = HIGH;
  static bool     stableState = HIGH;
  static uint32_t lastChange  = 0;

  bool reading = digitalRead(BUTTON_PIN);
  if (reading != lastReading) {
    lastChange  = millis();
    lastReading = reading;
  }
  if ((millis() - lastChange) > DEBOUNCE_MS && reading != stableState) {
    stableState = reading;
    if (stableState == LOW) {                      // pressed
      capturing = !capturing;
      ledSet(capturing);
      Serial.printf("[CAP] %s\n", capturing ? "RESUMED" : "PAUSED");
      if (capturing) {
        lastFrameMs = millis() - FRAME_INTERVAL_MS; // grab a frame promptly
      }
    }
  }
}

// ----------------------------------------------------------------------------
//  Setup / loop
// ----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n[BOOT] VoiceLink capture device starting...");

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  ledInit();

  if (!psramFound()) {
    Serial.println("[WARN] PSRAM not detected — set Tools -> PSRAM: \"OPI PSRAM\".");
  }

  cameraReady = cameraInit();
  if (cameraReady) Serial.println("[CAM] init OK (VGA JPEG)");

  // SoftAP — the device is its own Wi-Fi network.
  WiFi.mode(WIFI_AP);
  bool apOk = WiFi.softAP(AP_SSID, AP_PASSWORD, AP_CHANNEL);
  String ip = WiFi.softAPIP().toString();
  Serial.printf("[AP] %s  ssid=\"%s\"  pass=\"%s\"  ws=ws://%s:%u/\n",
                apOk ? "up" : "FAILED", AP_SSID, AP_PASSWORD, ip.c_str(), WS_PORT);

  frameSendBuf = (uint8_t*)ps_malloc(FRAME_SEND_BUF_MAX);
  if (!frameSendBuf) frameSendBuf = (uint8_t*)malloc(FRAME_SEND_BUF_MAX);
  if (!frameSendBuf) Serial.println("[WARN] could not allocate frame buffer");

  wsMutex = xSemaphoreCreateMutex();
  webSocket.begin();
  webSocket.onEvent(onWsEvent);

  if (micInit()) Serial.println("[MIC] I2S up (16 kHz mono, SPH0645)");
  else           Serial.println("[MIC] I2S init FAILED");

  ledSet(capturing);
  lastFrameMs = millis() - FRAME_INTERVAL_MS;       // grab the first frame asap

  // Audio read+send: high priority, pinned to core 0.
  xTaskCreatePinnedToCore(audioTask, "audio", 4096, NULL, 5, NULL, 0);

  Serial.println("[RDY] streaming. Press the button (GPIO14) to pause/resume.");
}

void loop() {  // runs on core 1
  if (xSemaphoreTake(wsMutex, portMAX_DELAY) == pdTRUE) {
    webSocket.loop();
    xSemaphoreGive(wsMutex);
  }

  serviceButton();

  if (capturing && cameraReady && (millis() - lastFrameMs >= FRAME_INTERVAL_MS)) {
    lastFrameMs = millis();
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("[CAM] capture failed");
    } else {
      if (frameSendBuf && (fb->len + 1) <= FRAME_SEND_BUF_MAX) {
        frameSendBuf[0] = MSG_FRAME;
        memcpy(frameSendBuf + 1, fb->buf, fb->len);
        if (xSemaphoreTake(wsMutex, portMAX_DELAY) == pdTRUE) {
          webSocket.broadcastBIN(frameSendBuf, fb->len + 1);
          xSemaphoreGive(wsMutex);
        }
      } else {
        Serial.printf("[CAM] frame too big to send: %u bytes\n", fb->len);
      }
      esp_camera_fb_return(fb);
    }
  }

  delay(1);  // yield to the IDLE/WiFi tasks
}
