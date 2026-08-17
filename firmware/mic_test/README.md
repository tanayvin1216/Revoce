# I2S microphone bench

A throwaway rig for proving the mic wiring works before trusting audio from it.
Separate from `firmware/voicelink_capture/` — nothing here ships.

## One-time setup

Install the **PlatformIO IDE** extension in VS Code (it bundles the `pio` CLI;
nothing else is required). Then open *this folder* — `firmware/mic_test` — as the
VS Code workspace root, not the repo root. PlatformIO looks for `platformio.ini`
in the folder you open.

## Board and mic

Set to the **WROVER** (classic ESP32). The S3 profile stays in `platformio.ini`
for when you switch back — change the two lines in `[active]`:

```ini
[active]
board = ${wrover.board}                      ; or ${s3.board}
flags = ${wrover.flags} ${sph0645.flags}     ; or ${s3.flags} / ${inmp441.flags}
```

Nothing here is interchangeable: GPIO 42 doesn't exist on the classic ESP32,
GPIO 33 collides with OPI PSRAM on the S3, and the SPH0645 needs a register
timing fix on the classic ESP32 that the INMP441 does not.

You don't have to know which mic you have. `mic_test` changes both mic-specific
settings live over serial, so you can find the working combination in one flash:

| Key | Effect |
|---|---|
| `b` | cycle bit shift — 14 (SPH0645) ↔ 16 (INMP441) |
| `s` | toggle the SPH0645 timing fix (classic ESP32 only) |

If you want to identify it by eye anyway, read the pin labels:
`SEL`/`LRCL`/`BCLK`/`DOUT` is an **SPH0645**; `L/R`/`WS`/`SCK`/`SD` is an
**INMP441**.

Whichever combination gives clean speech, copy it back into `[active]` before
building the recorder — that sketch is compile-time only.

## Step 1 — prove the wiring

```bash
pio run -e mic_test -t upload
pio device monitor -e mic_test
```

Read the output like this:

| What you see | What it means |
|---|---|
| `NO DATA — every sample identical` | Wiring fault. Data line delivered nothing. Check SD/SCK/WS and that L/R is tied to GND, not floating. |
| `quiet` with nothing else, ever | Mic is alive but you are below the noise gate — press `+` to raise gain. |
| `-40..-20 dB` when you talk | Correct. This is the target. |
| `CLIP` during normal speech | Too sensitive. Press `-`. |
| Level jumping between 0 and huge | Intermittent contact. Fix the connection before anything else. |

Adjust everything live in the monitor — no reflashing:

- `+` double the gain, `-` halve the gain
- `b` cycle the bit shift, `s` toggle the SPH0645 timing fix
- `g` print the current config

If the meter looks like noise no matter what you say, try `s` then `b` — a wrong
mic profile reads as constant hash rather than silence. Note the config line
that finally gives clean speech; you'll need all three values for the recorder.

## Step 2 — record something you can listen to

Put that gain into `MIC_GAIN` in `src/recorder.cpp`, then:

```bash
pio run -e recorder -t upload
```

**Close the serial monitor.** Only one program can hold the port.

```bash
python3 record.py
afplay mic.wav
```

`record.py` auto-detects the board, waits for it to boot, triggers a 5-second
take, and writes 16 kHz mono 16-bit WAV — which is exactly the format
Whisper-class transcription wants, so the file is ready to hand off as-is.

If the transfer is unreliable, set `SERIAL_BAUD` to `115200` in
`src/recorder.cpp` and run `python3 record.py -b 115200`. Slower, but works on
every USB-serial chip.
