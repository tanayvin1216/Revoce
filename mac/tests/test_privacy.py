"""Privacy / data-management tests: purge, pseudonymize, status, retention,
and the two server routes (status + software pause)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

import privacy
import server
from perception.wearable_client import CAPTURE_PAUSED

PROFILE = {
    "identity": {"name": "Jordan"},
    "relationships": [{"name": "Sam", "role": "partner", "aliases": ["sammy"]}],
}


def _write_taps(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _tap(ts: float, transcript: str = "Sam said hi", scene: str = "Jordan at a table") -> dict:
    return {
        "server_recv_ts": ts,
        "word": "yes",
        "board_id": "b1",
        "peer": "192.168.1.9",
        "context_snapshot": {
            "scene": scene,
            "transcript": transcript,
            "context": {"last_utterance": transcript, "active_question": None,
                        "recent_transcript": transcript, "personal_context": PROFILE},
        },
    }


# ----------------------------------------------------------------- status ---
def test_status_counts_and_timestamps(tmp_path):
    taps = tmp_path / "taps.jsonl"
    _write_taps(taps, [_tap(100.0), _tap(200.0), _tap(150.0)])
    profile = tmp_path / "personal_context.json"
    profile.write_text(json.dumps(PROFILE))

    s = privacy.status(taps, profile, repo_root=tmp_path)

    assert s["taps"]["count"] == 3
    assert s["taps"]["oldest_ts"] == 100.0 and s["taps"]["newest_ts"] == 200.0
    assert s["taps"]["bytes"] == taps.stat().st_size
    assert s["personal_context"]["exists"] is True
    assert s["no_media_artifacts"] is True
    assert s["facial_recognition"] is False and s["audio_video_persisted"] is False


def test_status_missing_files(tmp_path):
    s = privacy.status(tmp_path / "none.jsonl", tmp_path / "none.json", repo_root=tmp_path)
    assert s["taps"]["count"] == 0 and s["taps"]["exists"] is False
    assert s["taps"]["oldest_iso"] is None


def test_media_scan_flags_raw_media_outside_allowed_dirs(tmp_path):
    (tmp_path / "symbols").mkdir()
    (tmp_path / "symbols" / "yes.png").write_bytes(b"x")       # allowed
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "plot.png").write_bytes(b"x")      # allowed
    (tmp_path / "leak").mkdir()
    (tmp_path / "leak" / "frame.jpg").write_bytes(b"x")        # NOT allowed
    (tmp_path / "utt.wav").write_bytes(b"x")                   # NOT allowed

    found = privacy.scan_media_artifacts(tmp_path)

    assert found == ["leak/frame.jpg", "utt.wav"]


def test_real_repo_has_no_media_artifacts():
    """The live guarantee: no frame/audio was ever persisted in this checkout."""
    assert privacy.scan_media_artifacts(privacy.REPO_DIR) == []


# ------------------------------------------------------------------ purge ---
def test_purge_all_removes_file(tmp_path):
    taps = tmp_path / "taps.jsonl"
    _write_taps(taps, [_tap(1.0), _tap(2.0)])

    assert privacy.purge_taps(taps) == {"removed": 2, "kept": 0}
    assert not taps.exists()
    assert privacy.purge_taps(taps) == {"removed": 0, "kept": 0}   # idempotent


def test_purge_older_than_keeps_recent_and_drops_untimestamped(tmp_path):
    taps = tmp_path / "taps.jsonl"
    now = 1_000_000.0
    old = now - 10 * 86400
    fresh = now - 1 * 86400
    _write_taps(taps, [_tap(old), _tap(fresh), {"word": "no-ts"}])
    taps.open("a").write("not json\n")

    result = privacy.purge_taps(taps, older_than_days=7, now=now)

    assert result == {"removed": 3, "kept": 1}
    kept = [json.loads(l) for l in taps.read_text().splitlines()]
    assert kept[0]["server_recv_ts"] == fresh
    assert not taps.with_suffix(".jsonl.tmp").exists()


def test_apply_retention_default_is_keep(tmp_path, monkeypatch):
    taps = tmp_path / "taps.jsonl"
    _write_taps(taps, [_tap(1.0)])
    monkeypatch.delenv("VOICELINK_TAP_RETENTION_DAYS", raising=False)
    assert privacy.apply_retention(taps) is None
    assert taps.read_text().count("\n") == 1

    monkeypatch.setenv("VOICELINK_TAP_RETENTION_DAYS", "garbage")
    assert privacy.apply_retention(taps) is None

    monkeypatch.setenv("VOICELINK_TAP_RETENTION_DAYS", "30")
    assert privacy.apply_retention(taps) == {"removed": 1, "kept": 0}


# ----------------------------------------------------------- pseudonymize ---
def test_pseudonymize_replaces_names_stably_and_strips_profile(tmp_path):
    taps = tmp_path / "taps.jsonl"
    _write_taps(taps, [
        _tap(1.0, transcript="Hey Sammy, is Sam coming? Jordan wants sam.", scene="Jordan and Sam at a table"),
        _tap(2.0, transcript="sam again"),
    ])
    profile = tmp_path / "personal_context.json"
    profile.write_text(json.dumps(PROFILE))
    out = tmp_path / "taps.pseudo.jsonl"

    result = privacy.pseudonymize(taps, out, profile)

    assert result["records"] == 2
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    snap = recs[0]["context_snapshot"]
    assert snap["transcript"] == "Hey PERSON_2, is PERSON_2 coming? PERSON_1 wants PERSON_2."
    assert snap["scene"] == "PERSON_1 and PERSON_2 at a table"
    assert snap["context"]["last_utterance"].startswith("Hey PERSON_2")
    assert "personal_context" not in snap["context"]          # never exported
    assert recs[0]["peer"] == "REDACTED"
    assert recs[1]["context_snapshot"]["transcript"] == "PERSON_2 again"   # stable mapping
    assert "Sam" not in out.read_text() and "Jordan" not in out.read_text()


def test_pseudonymize_drop_transcripts(tmp_path):
    taps = tmp_path / "taps.jsonl"
    _write_taps(taps, [_tap(1.0, transcript="Sam says secret things")])
    profile = tmp_path / "personal_context.json"
    profile.write_text(json.dumps(PROFILE))
    out = tmp_path / "out.jsonl"

    privacy.pseudonymize(taps, out, profile, drop_transcripts=True)

    snap = json.loads(out.read_text())["context_snapshot"]
    assert snap["transcript"] == "[REDACTED]"
    assert snap["context"]["recent_transcript"] == "[REDACTED]"
    assert snap["scene"] == "PERSON_1 at a table"            # scene kept, names mapped


def test_pseudonymize_without_profile_still_writes(tmp_path):
    taps = tmp_path / "taps.jsonl"
    _write_taps(taps, [_tap(1.0)])
    out = tmp_path / "out.jsonl"
    result = privacy.pseudonymize(taps, out, tmp_path / "missing.json")
    assert result["records"] == 1 and result["names_mapped"] == 0


# ---------------------------------------------------------------- consent ---
def test_export_consent_template_refuses_overwrite(tmp_path):
    d = tmp_path / "consent"
    path = privacy.export_consent_template(d)
    assert path.exists() and "PROXY CONSENT" in path.read_text()
    with pytest.raises(FileExistsError):
        privacy.export_consent_template(d)
    privacy.export_consent_template(d, force=True)


# -------------------------------------------------------------------- CLI ---
def test_cli_status_and_purge_yes(tmp_path, capsys):
    taps = tmp_path / "taps.jsonl"
    _write_taps(taps, [_tap(1.0)])
    assert privacy.main(["status", "--taps", str(taps)]) in (0, 1)
    assert json.loads(capsys.readouterr().out)["taps"]["count"] == 1
    assert privacy.main(["purge", "--taps", "--path", str(taps), "--yes"]) == 0
    assert not taps.exists()


# ----------------------------------------------------------------- routes ---
class _Req:
    def __init__(self):
        self.app = {"state": {}}


def test_privacy_status_route_reports_pause_flag(monkeypatch, tmp_path):
    taps = tmp_path / "taps.jsonl"
    _write_taps(taps, [_tap(time.time())])
    monkeypatch.setattr(server, "TAP_LOG_PATH", taps)
    monkeypatch.setattr(server, "PROFILE_PATH", tmp_path / "personal_context.json")
    CAPTURE_PAUSED.clear()

    resp = asyncio.run(server.privacy_status_handler(_Req()))
    body = json.loads(resp.text)
    assert body["taps"]["count"] == 1
    assert body["capture_paused"] is False
    assert body["facial_recognition"] is False


def test_pause_route_toggles_flag_the_wearable_client_reads():
    CAPTURE_PAUSED.clear()
    try:
        r1 = json.loads(asyncio.run(server.privacy_pause_handler(_Req())).text)
        assert r1 == {"capture_paused": True} and CAPTURE_PAUSED.is_set()
        r2 = json.loads(asyncio.run(server.privacy_pause_handler(_Req())).text)
        assert r2 == {"capture_paused": False} and not CAPTURE_PAUSED.is_set()
    finally:
        CAPTURE_PAUSED.clear()


def test_routes_registered():
    app = server.make_app()
    paths = {(r.method, r.resource.canonical) for r in app.router.routes() if r.resource is not None}
    assert ("GET", "/api/privacy/status") in paths
    assert ("POST", "/api/privacy/pause") in paths
