"""
VoiceLink privacy / data-management tools — library + CLI, dependency-free.

What the system retains (see docs/RESPONSIBLE_AI.md for the full audit):
  * mac/taps.jsonl            — every vocabulary tap + the TEXT context snapshot
                                (scene sentence, transcript, flags) that produced
                                its board. Never audio, never frames.
  * mac/personal_context.json — the user's personal-context store. Stays on
                                the Mac; gitignored; never sent anywhere.
  * audio / video             — ephemeral only (RAM buffers in perception);
                                nothing is ever written to disk. `status`
                                verifies that by scanning the repo.

Commands:
  python3 privacy.py status
  python3 privacy.py purge --taps [--older-than DAYS] [--yes]
  python3 privacy.py pseudonymize --in taps.jsonl --out taps.pseudo.jsonl [--drop-transcripts]
  python3 privacy.py export-consent-template

Library entry points (used by server.py): status(), purge_taps(), pseudonymize().
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

MAC_DIR = Path(__file__).resolve().parent
REPO_DIR = MAC_DIR.parent
TAP_LOG_PATH = MAC_DIR / "taps.jsonl"
PROFILE_PATH = MAC_DIR / "personal_context.json"
CONSENT_DIR = REPO_DIR / "docs" / "consent"

# Raw-media extensions that must NEVER appear as runtime artifacts. Pictograms
# (symbols/) and evaluation plots (figures/) are the only legitimate images.
MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".wav", ".pcm", ".mp4", ".webm", ".m4a", ".flac", ".raw"}
# "clips" is mac/intel/clips/: two SYNTHESIZED text-to-speech benchmark
# sentences committed so every host times whisper on identical audio. They are
# not captured from any person; nothing recorded by the wearable is ever
# written to disk (that is what this scan enforces).
MEDIA_ALLOWED_DIRS = ("symbols", "figures", "clips")
SCAN_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build"}

# Tap-record fields that can carry free text (and therefore names).
_TEXT_FIELDS = ("scene", "transcript")
_CONTEXT_TEXT_KEYS = ("last_utterance", "active_question", "recent_transcript")


# ----------------------------------------------------------------------------
#  status
# ----------------------------------------------------------------------------
def _iter_taps(path: Path):
    """Yield (line, record_or_None) for every line of a taps.jsonl file."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                yield line, json.loads(line)
            except json.JSONDecodeError:
                yield line, None


def scan_media_artifacts(root: Path = REPO_DIR) -> list[str]:
    """Return repo-relative paths of raw media files OUTSIDE symbols/ and figures/.
    An empty list is the evidence that no frame/audio was ever persisted."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SCAN_SKIP_DIRS]
        rel_parts = Path(dirpath).relative_to(root).parts
        if any(p in MEDIA_ALLOWED_DIRS for p in rel_parts):
            continue
        for name in filenames:
            if Path(name).suffix.lower() in MEDIA_EXTENSIONS:
                found.append(str(Path(dirpath, name).relative_to(root)))
    return sorted(found)


def status(tap_path: Path = TAP_LOG_PATH, profile_path: Path = PROFILE_PATH,
           repo_root: Path = REPO_DIR) -> dict:
    """Machine-readable privacy status (also served at /api/privacy/status)."""
    count = 0
    oldest: float | None = None
    newest: float | None = None
    for _, rec in _iter_taps(tap_path):
        count += 1
        if rec is None:
            continue
        ts = rec.get("server_recv_ts")
        if isinstance(ts, (int, float)):
            oldest = ts if oldest is None else min(oldest, ts)
            newest = ts if newest is None else max(newest, ts)
    artifacts = scan_media_artifacts(repo_root)
    return {
        "taps": {
            "path": str(tap_path),
            "exists": tap_path.exists(),
            "count": count,
            "bytes": tap_path.stat().st_size if tap_path.exists() else 0,
            "oldest_ts": oldest,
            "newest_ts": newest,
            "oldest_iso": _iso(oldest),
            "newest_iso": _iso(newest),
        },
        "personal_context": {
            "path": str(profile_path),
            "exists": profile_path.exists(),
            "bytes": profile_path.stat().st_size if profile_path.exists() else 0,
        },
        "retention_days": _retention_days_env(),
        "media_artifacts_on_disk": artifacts,
        "no_media_artifacts": not artifacts,
        "audio_video_persisted": False,   # by design: RAM-only buffers (see RESPONSIBLE_AI.md)
        "facial_recognition": False,      # the VLM writes one scene sentence; no face model exists
        "checked_ts": time.time(),
    }


def _retention_days_env() -> float:
    try:
        return float(os.environ.get("VOICELINK_TAP_RETENTION_DAYS", "0") or 0)
    except ValueError:
        return 0.0


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


# ----------------------------------------------------------------------------
#  purge
# ----------------------------------------------------------------------------
def purge_taps(tap_path: Path = TAP_LOG_PATH, older_than_days: float | None = None,
               now: float | None = None) -> dict:
    """Delete tap records. With older_than_days, keep only records newer than the
    cutoff (records without a parseable timestamp are treated as old). Without
    it, the whole file is removed. Writes via a temp file + atomic rename so a
    crash mid-purge can never leave a half-written log.

    Returns {"removed": n, "kept": n}.
    """
    if not tap_path.exists():
        return {"removed": 0, "kept": 0}
    if older_than_days is None:
        removed = sum(1 for _ in _iter_taps(tap_path))
        tap_path.unlink()
        return {"removed": removed, "kept": 0}

    cutoff = (now if now is not None else time.time()) - float(older_than_days) * 86400.0
    kept_lines: list[str] = []
    removed = 0
    for line, rec in _iter_taps(tap_path):
        ts = rec.get("server_recv_ts") if isinstance(rec, dict) else None
        if isinstance(ts, (int, float)) and ts >= cutoff:
            kept_lines.append(line)
        else:
            removed += 1
    tmp = tap_path.with_suffix(tap_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    os.replace(tmp, tap_path)
    return {"removed": removed, "kept": len(kept_lines)}


def apply_retention(tap_path: Path = TAP_LOG_PATH, env_var: str = "VOICELINK_TAP_RETENTION_DAYS") -> dict | None:
    """Server-startup hook: honour VOICELINK_TAP_RETENTION_DAYS (0/unset = keep
    everything, which is the default because tap data is irreplaceable)."""
    raw = os.environ.get(env_var, "0").strip()
    try:
        days = float(raw)
    except ValueError:
        print(f"[privacy] ignoring non-numeric {env_var}={raw!r}")
        return None
    if days <= 0:
        return None
    result = purge_taps(tap_path, older_than_days=days)
    print(f"[privacy] retention {days:g}d: removed {result['removed']} tap(s), kept {result['kept']}")
    return result


# ----------------------------------------------------------------------------
#  pseudonymize
# ----------------------------------------------------------------------------
def names_from_profile(profile: dict | None) -> list[str]:
    """Collect every personal name / alias in a personal-context profile."""
    if not isinstance(profile, dict):
        return []
    names: list[str] = []
    identity = profile.get("identity") or {}
    if isinstance(identity, dict) and identity.get("name"):
        names.append(str(identity["name"]))
    for rel in profile.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        if rel.get("name"):
            names.append(str(rel["name"]))
        for alias in rel.get("aliases") or []:
            names.append(str(alias))
    # de-dup case-insensitively, keep first spelling, longest first so "Sammy"
    # is replaced before "Sam" can clip it.
    seen: dict[str, str] = {}
    for n in names:
        if n.strip() and n.lower() not in seen:
            seen[n.lower()] = n.strip()
    return sorted(seen.values(), key=len, reverse=True)


class Pseudonymizer:
    """Stable name -> PERSON_n mapping. The same name (any case) always maps to
    the same token within one run; aliases of one relationship share a token."""

    def __init__(self, profile: dict | None) -> None:
        self._map: dict[str, str] = {}
        counter = 0
        identity = (profile or {}).get("identity") or {}
        if isinstance(identity, dict) and identity.get("name"):
            counter += 1
            self._map[str(identity["name"]).lower()] = f"PERSON_{counter}"
        for rel in (profile or {}).get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            counter += 1
            token = f"PERSON_{counter}"
            for n in [rel.get("name"), *(rel.get("aliases") or [])]:
                if n and str(n).lower() not in self._map:
                    self._map[str(n).lower()] = token
        names = sorted(self._map, key=len, reverse=True)
        self._pattern = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE
        ) if names else None

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._map)

    def text(self, value):
        if not isinstance(value, str) or self._pattern is None:
            return value
        return self._pattern.sub(lambda m: self._map[m.group(0).lower()], value)

    def record(self, rec: dict, drop_transcripts: bool = False) -> dict:
        out = json.loads(json.dumps(rec))  # deep copy, JSON-safe
        snap = out.get("context_snapshot")
        if isinstance(snap, dict):
            for key in _TEXT_FIELDS:
                if key in snap:
                    snap[key] = self._redact(snap[key], drop_transcripts and key == "transcript")
            ctx = snap.get("context")
            if isinstance(ctx, dict):
                for key in _CONTEXT_TEXT_KEYS:
                    if key in ctx:
                        ctx[key] = self._redact(ctx[key], drop_transcripts)
                # The personal-context store must never leave the device, so it
                # is never included in an export, even pseudonymized.
                ctx.pop("personal_context", None)
        if out.get("peer"):
            out["peer"] = "REDACTED"   # client IP is not needed for analysis
        return out

    def _redact(self, value, drop: bool):
        if value is None:
            return None
        if drop:
            return "[REDACTED]"
        return self.text(value)


def pseudonymize(in_path: Path, out_path: Path, profile_path: Path = PROFILE_PATH,
                 drop_transcripts: bool = False) -> dict:
    """Write a shareable copy of a tap log with names replaced by PERSON_n."""
    profile = None
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        profile = None
    ps = Pseudonymizer(profile)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for _, rec in _iter_taps(in_path):
            if rec is None:
                continue  # drop unparseable lines rather than leak them verbatim
            f.write(json.dumps(ps.record(rec, drop_transcripts)) + "\n")
            n += 1
    return {"records": n, "names_mapped": len(ps.mapping), "out": str(out_path)}


# ----------------------------------------------------------------------------
#  consent template
# ----------------------------------------------------------------------------
CONSENT_TEMPLATE = """\
VOICELINK — PROXY CONSENT FOR PARTICIPATION (TEMPLATE)
======================================================

Project:      VoiceLink — wearable context-aware AAC vocabulary support
Participant:  ______________________________   Date of birth: ____________
Proxy/guardian (if participant cannot consent for themselves):
              ______________________________   Relationship: _____________
Study contact: _____________________________   Email/phone: ______________

WHAT THE DEVICE DOES
  A small body-worn device streams camera frames and microphone audio to ONE
  computer in the same room over a private Wi-Fi link. That computer turns them
  into (a) a one-sentence description of the scene and (b) a text transcript of
  nearby speech, and uses those to suggest words on the participant's AAC board.

WHAT IS RECORDED AND KEPT
  * Audio and video are processed in memory only and are NEVER saved to disk.
  * When the participant taps a word, the word and the TEXT context at that
    moment (scene sentence, transcript snippet) are written to a log on that
    computer. This log is kept to improve the system and is never uploaded.
  * The participant's personal-context profile (names of people close to them,
    routines, today's events) stays on that computer only.
  * No face recognition or person identification is performed.
  * Logs can be pseudonymized (names replaced by PERSON_1, PERSON_2 ...) before
    any analysis, and can be deleted at any time on request
    (python3 mac/privacy.py purge --taps).

PEOPLE NEARBY
  Speech from people near the participant may appear in transcripts. A red LED
  on the device is lit whenever capture is active, and a physical button pauses
  capture immediately. Bystanders will be told the device is active.

YOUR RIGHTS
  * Participation is voluntary; you can stop at any time with no consequence.
  * You may ask to see, pseudonymize, or delete all logged data.
  * Data is used only for the purposes above and is not shared with third
    parties or cloud services.

CONSENT
  I have read the above (or had it explained to me), my questions have been
  answered, and I consent on behalf of the participant named above.

  Proxy signature: ____________________________   Date: ______________
  Participant assent (where possible): ______________   Date: __________
  Researcher signature: _______________________   Date: ______________

NOTE: this is a template. Review with your institution's ethics/IRB process
and local law (e.g. GDPR Art. 9 / COPPA / state wiretap rules) before use.
"""


def export_consent_template(consent_dir: Path = CONSENT_DIR, force: bool = False) -> Path:
    target = consent_dir / "PROXY_CONSENT_TEMPLATE.txt"
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists (use --force to overwrite)")
    consent_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(CONSENT_TEMPLATE, encoding="utf-8")
    return target


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------
def _cmd_status(args) -> int:
    s = status(Path(args.taps))
    print(json.dumps(s, indent=2))
    return 0 if s["no_media_artifacts"] else 1


def _cmd_purge(args) -> int:
    if not args.taps:
        print("nothing selected; pass --taps", file=sys.stderr)
        return 2
    path = Path(args.path)
    what = f"taps older than {args.older_than} day(s)" if args.older_than is not None else "ALL taps"
    if not args.yes:
        answer = input(f"Delete {what} from {path}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return 1
    print(json.dumps(purge_taps(path, args.older_than)))
    return 0


def _cmd_pseudonymize(args) -> int:
    print(json.dumps(pseudonymize(Path(args.inp), Path(args.out), Path(args.profile),
                                  drop_transcripts=args.drop_transcripts)))
    return 0


def _cmd_consent(args) -> int:
    try:
        path = export_consent_template(Path(args.dir), force=args.force)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="what is retained, where, and proof no media is on disk")
    s.add_argument("--taps", default=str(TAP_LOG_PATH))
    s.set_defaults(fn=_cmd_status)

    g = sub.add_parser("purge", help="delete tap records")
    g.add_argument("--taps", action="store_true", help="target the tap log")
    g.add_argument("--path", default=str(TAP_LOG_PATH))
    g.add_argument("--older-than", type=float, default=None, metavar="DAYS")
    g.add_argument("--yes", action="store_true", help="skip confirmation")
    g.set_defaults(fn=_cmd_purge)

    z = sub.add_parser("pseudonymize", help="copy a tap log with names -> PERSON_n")
    z.add_argument("--in", dest="inp", default=str(TAP_LOG_PATH))
    z.add_argument("--out", required=True)
    z.add_argument("--profile", default=str(PROFILE_PATH))
    z.add_argument("--drop-transcripts", action="store_true")
    z.set_defaults(fn=_cmd_pseudonymize)

    c = sub.add_parser("export-consent-template", help="write docs/consent/PROXY_CONSENT_TEMPLATE.txt")
    c.add_argument("--dir", default=str(CONSENT_DIR))
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=_cmd_consent)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
