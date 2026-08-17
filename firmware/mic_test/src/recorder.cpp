// ---------------------------------------------------------------------------
//  VoiceLink — I2S microphone recorder
//
//  Records RECORD_SECONDS of audio into PSRAM, then dumps it as raw 16-bit
//  mono PCM over USB. Recording to memory first means a slow serial link can
//  never corrupt the audio itself.
//
//  Protocol (record.py on the Mac speaks the other half):
//      board -> "READY"          waiting for a command
//      host  -> 'r'              start
//      board -> "RECORDING"      speak now
//      board -> "LEN:<bytes>"    payload size, then <bytes> of raw PCM
//
//  Set MIC_GAIN below to whatever value you settled on in mic_test.
//  Pins and I2S port come from platformio.ini — same wiring as mic_test.
// ---------------------------------------------------------------------------

#include <Arduino.h>
#include <driver/i2s.h>
#include <esp_heap_caps.h>

#if defined(MIC_SPH0645) && defined(CONFIG_IDF_TARGET_ESP32)
#include <soc/i2s_reg.h>
#endif

static const i2s_port_t MIC_PORT = (i2s_port_t)MIC_I2S_PORT;

static const int SAMPLE_RATE    = 16000;
static const int RECORD_SECONDS = 5;
static const int BUF_SAMPLES    = 256;

// Match this to the gain you found with mic_test.
static const float MIC_GAIN = 8.0f;

// Must match BAUD in record.py. Drop both to 115200 if the transfer is flaky.
static const unsigned long SERIAL_BAUD = 921600;

static int16_t *pcm = nullptr;
static size_t   recordSamples = 0;
static int32_t  rawBuffer[BUF_SAMPLES];
static float    dcOffset = 0.0f;

static bool micInit() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,   // L/R tied to GND
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BUF_SAMPLES,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pins = {
    .mck_io_num   = I2S_PIN_NO_CHANGE,
    .bck_io_num   = MIC_SCK_PIN,
    .ws_io_num    = MIC_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = MIC_SD_PIN
  };

  if (i2s_driver_install(MIC_PORT, &cfg, 0, NULL) != ESP_OK) return false;
  if (i2s_set_pin(MIC_PORT, &pins) != ESP_OK) return false;

#if defined(MIC_SPH0645) && defined(CONFIG_IDF_TARGET_ESP32)
  // See mic_test.cpp — SPH0645 timing fix for the classic ESP32 receiver.
  REG_SET_BIT(I2S_TIMING_REG(MIC_PORT), BIT(9));
  REG_SET_BIT(I2S_CONF_REG(MIC_PORT), I2S_RX_MSB_SHIFT);
#endif

  i2s_zero_dma_buffer(MIC_PORT);
  return true;
}

static int16_t convertSample(int32_t raw) {
  const float s = (float)(raw >> MIC_SHIFT);
  dcOffset += (s - dcOffset) * 0.002f;

  float v = (s - dcOffset) * MIC_GAIN;
  if (v >  32767.0f) v =  32767.0f;
  if (v < -32768.0f) v = -32768.0f;
  return (int16_t)v;
}

// Run the converter over ~0.5 s of discarded audio so dcOffset has settled
// before the samples we keep.
static void warmUp() {
  size_t bytesRead = 0;
  for (int block = 0; block < (SAMPLE_RATE / 2) / BUF_SAMPLES; block++) {
    i2s_read(MIC_PORT, rawBuffer, sizeof(rawBuffer), &bytesRead, portMAX_DELAY);
    const int count = bytesRead / sizeof(int32_t);
    for (int i = 0; i < count; i++) convertSample(rawBuffer[i]);
  }
}

static void recordAndDump() {
  warmUp();
  Serial.println("RECORDING");

  size_t written = 0;
  while (written < recordSamples) {
    size_t bytesRead = 0;
    i2s_read(MIC_PORT, rawBuffer, sizeof(rawBuffer), &bytesRead, portMAX_DELAY);
    const int count = bytesRead / sizeof(int32_t);
    for (int i = 0; i < count && written < recordSamples; i++) {
      pcm[written++] = convertSample(rawBuffer[i]);
    }
  }

  Serial.printf("LEN:%u\n", (unsigned)(recordSamples * sizeof(int16_t)));
  Serial.flush();
  Serial.write((const uint8_t *)pcm, recordSamples * sizeof(int16_t));
  Serial.flush();
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(800);

  recordSamples = (size_t)SAMPLE_RATE * RECORD_SECONDS;
  pcm = (int16_t *)ps_malloc(recordSamples * sizeof(int16_t));

  if (pcm == nullptr) {   // no PSRAM — fall back to a shorter internal-RAM take
    recordSamples = (size_t)SAMPLE_RATE * 2;
    pcm = (int16_t *)malloc(recordSamples * sizeof(int16_t));
  }
  if (pcm == nullptr) {
    Serial.println("FATAL: could not allocate an audio buffer");
    while (true) delay(1000);
  }

  if (!micInit()) {
    Serial.println("FATAL: i2s driver failed to start");
    while (true) delay(1000);
  }

  Serial.printf("buffer: %.1f seconds @ %d Hz\n",
                (float)recordSamples / SAMPLE_RATE, SAMPLE_RATE);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() && Serial.read() == 'r') {
    recordAndDump();
    Serial.println("READY");
  }
}
