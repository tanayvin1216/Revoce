// ---------------------------------------------------------------------------
//  VoiceLink — I2S microphone level meter
//
//  Proves the mic wiring is good before you trust any audio from it. Prints a
//  real dBFS level: -90 is silence, 0 is clipping. Speech at ~30 cm should land
//  somewhere in the -40..-20 range once the gain is set.
//
//  Everything that depends on WHICH mic you have is adjustable live, so you can
//  find the working combination in one flash instead of guessing:
//      +   double the gain (more sensitive)
//      -   halve the gain  (less sensitive)
//      b   cycle the bit shift: 14 (SPH0645) <-> 16 (INMP441)
//      s   toggle the SPH0645 timing fix (classic ESP32 only)
//      g   print the current config
//
//  Whichever combination gives clean speech, copy it back into platformio.ini
//  before flashing the recorder.
//
//  Pins and I2S port come from platformio.ini. Wiring for the WROVER profile:
//      VDD / 3V   -> 3.3V   (3.3V ONLY — 5V can kill the part)
//      GND        -> GND
//      L/R or SEL -> GND    (ties the mic to the LEFT slot; must not float)
//      WS  / LRCL -> GPIO 32
//      SCK / BCLK -> GPIO 14
//      SD  / DOUT -> GPIO 33
// ---------------------------------------------------------------------------

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>

// The SPH0645 clocks its data out one cycle later than the classic ESP32's I2S
// receiver expects; without the fix its samples decode as noise. The S3 does
// not need it, and the registers differ there, so it compiles out.
#if defined(CONFIG_IDF_TARGET_ESP32)
#define SPH_FIX_AVAILABLE 1
#include <soc/i2s_reg.h>
#endif

static const i2s_port_t MIC_PORT = (i2s_port_t)MIC_I2S_PORT;

static const int   SAMPLE_RATE = 16000;
static const int   BUF_SAMPLES = 256;
static const float FULL_SCALE  = 32768.0f;   // full scale of a 16-bit sample

// Anything below this prints as "quiet". Make it less negative (e.g. -45) to
// ignore more room noise.
static const float NOISE_GATE_DB = -55.0f;

// --- Live-tunable config, seeded from platformio.ini ------------------------

// Makeup gain. 1 = faithful but very quiet, 8 = normal speech at ~30 cm,
// 32 = picks up the whole room.
static float micGain = 8.0f;

// How far to shift the raw 32-bit I2S word down to a 16-bit sample.
static int shiftBits = MIC_SHIFT;

#if defined(MIC_SPH0645)
static bool sphFix = true;
#else
static bool sphFix = false;
#endif

// The SPH0645 in particular carries a large DC bias; left in, it reads as
// permanent loudness. This tracks and subtracts it.
static float dcOffset = 0.0f;

static int32_t rawBuffer[BUF_SAMPLES];
static bool    micRunning = false;

// If the line is dead from the moment we start, run the pin test on our own
// rather than waiting for someone to press a key.
static int  deadBlocks = 0;
static bool autoDiagDone = false;

// Rolling record of how many recent blocks carried real data. Intermittent
// wiring shows up here as a percentage that moves when you wiggle a wire —
// which is how you find WHICH joint is bad.
static const int LINK_WINDOW = 40;
static bool linkHistory[LINK_WINDOW];
static int  linkIndex = 0;
static int  linkGood = 0;

static int updateLink(bool ok) {
  if (linkHistory[linkIndex]) linkGood--;
  linkHistory[linkIndex] = ok;
  if (ok) linkGood++;
  linkIndex = (linkIndex + 1) % LINK_WINDOW;
  return (linkGood * 100) / LINK_WINDOW;
}

// ---------------------------------------------------------------------------

// The mic needs two connections to ground (GND and SEL/L-R) but the board may
// only break out one GND pin. SEL is a high-impedance input drawing microamps,
// so a spare GPIO held LOW serves as a second ground perfectly well.
static void startSelGround() {
#if defined(MIC_SEL_GND_PIN)
  pinMode(MIC_SEL_GND_PIN, OUTPUT);
  digitalWrite(MIC_SEL_GND_PIN, LOW);
  Serial.printf("SEL ground: GPIO %d held LOW\n", MIC_SEL_GND_PIN);
#endif
}

static bool micInit() {
  if (micRunning) {
    i2s_driver_uninstall(MIC_PORT);
    micRunning = false;
  }

  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,   // L/R (or SEL) tied to GND
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

#if defined(SPH_FIX_AVAILABLE)
  if (sphFix) {
    REG_SET_BIT(I2S_TIMING_REG(MIC_PORT), BIT(9));
    REG_SET_BIT(I2S_CONF_REG(MIC_PORT), I2S_RX_MSB_SHIFT);
  }
#endif

  i2s_zero_dma_buffer(MIC_PORT);
  micRunning = true;
  dcOffset = 0.0f;
  return true;
}

// Let the DC tracker settle before we print anything, or the first second of
// output is meaningless.
static void warmUp() {
  size_t bytesRead = 0;
  for (int block = 0; block < (SAMPLE_RATE / 2) / BUF_SAMPLES; block++) {
    i2s_read(MIC_PORT, rawBuffer, sizeof(rawBuffer), &bytesRead, portMAX_DELAY);
    const int count = bytesRead / sizeof(int32_t);
    for (int i = 0; i < count; i++) {
      const float s = (float)(rawBuffer[i] >> shiftBits);
      dcOffset += (s - dcOffset) * 0.002f;
    }
  }
}

// Reads the data line as a plain GPIO with the internal pull-up, then the
// pull-down. What the line does under each tells you WHY there is no data:
//   follows the pull  -> nothing is driving it (mic dead, unpowered, or open joint)
//   stuck LOW         -> shorted to GND (solder bridge is the usual cause)
//   stuck HIGH        -> shorted to 3.3V
static void pinDiagnostic() {
  i2s_driver_uninstall(MIC_PORT);
  micRunning = false;

  pinMode(MIC_SD_PIN, INPUT_PULLUP);
  delay(20);
  const int pulledUp = digitalRead(MIC_SD_PIN);

  pinMode(MIC_SD_PIN, INPUT_PULLDOWN);
  delay(20);
  const int pulledDown = digitalRead(MIC_SD_PIN);

  // Repeat the wiring here too — the boot banner is easy to miss on the S3,
  // where the USB port re-enumerates before a monitor can attach.
  Serial.printf("\n--- wiring: WS=GPIO%d  SCK=GPIO%d  SD=GPIO%d  I2S port %d\n",
                MIC_WS_PIN, MIC_SCK_PIN, MIC_SD_PIN, MIC_I2S_PORT);
#if defined(MIC_SEL_GND_PIN)
  Serial.printf("    SEL ground: GPIO %d held LOW (reads %d, must be 0)\n",
                MIC_SEL_GND_PIN, digitalRead(MIC_SEL_GND_PIN));
#endif
  Serial.printf("--- data line (GPIO %d) ---\n", MIC_SD_PIN);
  Serial.printf("  with pull-up: %d   with pull-down: %d\n", pulledUp, pulledDown);

  if (pulledUp == 1 && pulledDown == 0) {
    Serial.println("  VERDICT: FLOATING — nothing is driving this line.");
    Serial.println("  The mic is not outputting. In order of likelihood:");
    Serial.println("   1. Cold joint on VDD or GND -> mic has no power.");
    Serial.println("      MEASURE 3.3V across the mic's OWN VDD and GND pins.");
    Serial.println("   2. Cold joint on this data pin -> reflow it.");
    Serial.println("   3. The mic is dead.");
  } else if (pulledUp == 0 && pulledDown == 0) {
    Serial.println("  VERDICT: STUCK LOW — this line is shorted to ground.");
    Serial.println("  That is a SOLDER BRIDGE, and it means YOUR MIC IS PROBABLY FINE.");
    Serial.println("  Find it: multimeter continuity, this pin against each neighbour.");
    Serial.println("  Clear it with solder braid + flux. Do not drag the iron around.");
  } else {
    Serial.println("  VERDICT: STUCK HIGH — shorted to 3.3V. Also a solder bridge;");
    Serial.println("  the mic is probably fine. Check this pin against VDD.");
  }
  Serial.println("---------------------------\n");

  // Drop the pull-down before handing the pin back to I2S. i2s_set_pin only
  // sets direction and the GPIO matrix — it leaves the pull resistor alone, so
  // without this the data line stays weakly tied low for every later reading.
  pinMode(MIC_SD_PIN, INPUT);

  micInit();
  warmUp();
}

static void printConfig() {
  Serial.printf(">>> gain %.2f | shift %d (%s) | sph_fix %s\n",
                micGain,
                shiftBits,
                shiftBits == 14 ? "SPH0645" : "INMP441",
                sphFix ? "ON" : "off");
}

static void handleSerialCommand() {
  if (!Serial.available()) return;
  const int c = Serial.read();

  if (c == '+') {
    micGain = min(micGain * 2.0f, 1024.0f);
  } else if (c == '-') {
    micGain = max(micGain / 2.0f, 0.25f);
  } else if (c == 'b') {
    shiftBits = (shiftBits == 14) ? 16 : 14;
  } else if (c == 'p') {
    pinDiagnostic();
    return;
  } else if (c == 's') {
#if defined(SPH_FIX_AVAILABLE)
    sphFix = !sphFix;
    micInit();       // the fix is a register write, so the driver must restart
    warmUp();
#else
    Serial.println(">>> sph fix is classic-ESP32 only; not available here");
    return;
#endif
  } else if (c != 'g') {
    return;
  }

  printConfig();
}

void setup() {
  Serial.begin(115200);
  delay(600);
  Serial.println();
  Serial.println("VoiceLink mic level meter");
  Serial.printf("pins: WS=%d SCK=%d SD=%d   I2S port %d\n",
                MIC_WS_PIN, MIC_SCK_PIN, MIC_SD_PIN, MIC_I2S_PORT);
  Serial.println("keys: '+'/'-' gain   'b' bit shift   's' sph fix   'p' pin test   'g' config");
  Serial.println("-----------------------------------------------------------");

  startSelGround();
  if (!micInit()) {
    Serial.println("FATAL: i2s driver failed to start");
    while (true) delay(1000);
  }
  printConfig();
  warmUp();
}

void loop() {
  handleSerialCommand();

  size_t bytesRead = 0;
  i2s_read(MIC_PORT, rawBuffer, sizeof(rawBuffer), &bytesRead, portMAX_DELAY);
  const int count = bytesRead / sizeof(int32_t);
  if (count == 0) return;

  double  sumSquares = 0.0;
  int32_t rawMin = INT32_MAX;
  int32_t rawMax = INT32_MIN;
  bool    clipped = false;
  int     railCount = 0;   // samples sitting exactly on a supply rail

  for (int i = 0; i < count; i++) {
    const int32_t raw = rawBuffer[i];
    if (raw < rawMin) rawMin = raw;
    if (raw > rawMax) rawMax = raw;
    if (raw == 0 || raw == -1) railCount++;

    const float s = (float)(raw >> shiftBits);
    dcOffset += (s - dcOffset) * 0.002f;
    const float centered = s - dcOffset;

    if (fabsf(centered) * micGain > 32767.0f) clipped = true;
    sumSquares += (double)centered * centered;
  }

  // Every sample bit-identical means the data line delivered nothing at all —
  // that is a wiring fault, not a quiet room.
  // Every sample bit-identical means this block carried nothing at all.
  const bool blockOk = (rawMin != rawMax);
  const int  link = updateLink(blockOk);

  if (!blockOk) {
    // 0x00000000 = line never leaves ground; 0xFFFFFFFF = it sits high.
    Serial.printf("link %3d%%  -- no data (0x%08X)\n", link, (unsigned)rawMin);

    // Re-run the pin test every ~5 s while the line stays dead. It has to
    // repeat: a one-shot at boot fires before anyone has attached a monitor.
    if (++deadBlocks >= 100) {
      deadBlocks = 0;
      Serial.println("\n>>> Line has been dead for 5 seconds. Pin test:");
      pinDiagnostic();
    }
    delay(50);
    return;
  }
  deadBlocks = 0;

  // A block can vary yet still be junk: when a loose wire flaps, the line jumps
  // between the rails and that square wave computes as an enormous "level".
  // Real audio never parks a quarter of its samples exactly on 0 or -1.
  const int railPercent = (railCount * 100) / count;
  if (railPercent > 25) {
    Serial.printf("link %3d%%  -- CONTACT NOISE (%d%% of samples on a rail), not audio\n",
                  link, railPercent);
    delay(50);
    return;
  }

  const float rms = sqrtf((float)(sumSquares / count)) * micGain;
  const float db  = 20.0f * log10f(rms / FULL_SCALE + 1e-9f);

  if (db < NOISE_GATE_DB) {
    Serial.printf("link %3d%%  quiet\n", link);
    delay(50);
    return;
  }

  int bars = (int)((db + 70.0f) * (40.0f / 70.0f));
  if (bars < 0)  bars = 0;
  if (bars > 40) bars = 40;

  Serial.printf("link %3d%% %6.1f dB %s ", link, db, clipped ? "CLIP" : "    ");
  for (int i = 0; i < bars; i++) Serial.print('#');
  Serial.println();

  delay(50);
}
