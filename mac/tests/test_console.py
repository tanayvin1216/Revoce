from __future__ import annotations

import asyncio
import json
from pathlib import Path

import server


class FakeRanker:
    def __init__(self) -> None:
        self.calls = 0

    def rank(self, system_prompt, user_prompt, max_tokens=None):
        self.calls += 1
        return [
            {"word": "yes", "symbol_keyword": "yes", "function": "comment"},
            {"word": "no", "symbol_keyword": "no", "function": "reject"},
            {"word": "chlorophyll", "symbol_keyword": "plant", "function": "share-information"},
            {"word": "how does it work", "symbol_keyword": "question", "function": "question"},
        ]


def _app_state(tmp_path: Path) -> dict:
    return {
        "ranker": FakeRanker(),
        "board_strategy": "diverse",
        "personal_context": {"identity": {"name": "Jordan"}},
        "phase_status_path": tmp_path / "phase_status.json",
    }


def test_phase_status_payload_reads_status_file(tmp_path):
    status_path = tmp_path / "phase_status.json"
    status_path.write_text(
        json.dumps({"phases": [{"id": 0, "status": "green"}]}),
        encoding="utf-8",
    )

    payload = server._phase_status_payload(status_path)

    assert payload["phases"][0]["status"] == "green"


def test_console_scenarios_include_phase_2_scripts():
    ids = {scenario["id"] for scenario in server._console_scenarios()}

    assert {"airport", "photosynthesis", "doctor"} <= ids


def test_console_run_calls_seam_once_and_captures_candidate_pool(monkeypatch, tmp_path):
    calls = []

    def fake_rank_vocabulary(scene, transcript, context, ranker, strategy, topics):
        calls.append({
            "scene": scene,
            "transcript": transcript,
            "context": context,
            "strategy": strategy,
            "topics": topics,
        })
        raw_pool = ranker.rank("system", "user", max_tokens=10)
        return [
            {"word": item["word"], "symbol": f"/symbols/{i}.png"}
            for i, item in enumerate(raw_pool[:3], start=1)
        ]

    monkeypatch.setattr(server, "rank_vocabulary", fake_rank_vocabulary)

    result = asyncio.run(server._run_console_payload(
        _app_state(tmp_path),
        {
            "scene": "a biology classroom with a whiteboard",
            "transcript": "How do plants make sugar during photosynthesis?",
            "profile_name": "Jordan",
        },
    ))

    assert len(calls) == 1
    assert calls[0]["scene"] == "a biology classroom with a whiteboard"
    assert calls[0]["strategy"] == "diverse"
    assert result["scene_description"] == "a biology classroom with a whiteboard"
    assert [item["word"] for item in result["candidate_pool"]] == [
        "yes",
        "no",
        "chlorophyll",
        "how does it work",
    ]
    assert [item["word"] for item in result["board"]] == ["yes", "no", "chlorophyll"]
    assert set(result["latency_ms"]) >= {"whisper", "vlm", "pool_gen", "selection", "total"}


def test_console_html_contains_required_controls():
    html = (server.WEB_DIR / "console.html").read_text(encoding="utf-8")

    for needle in [
        "dropzone",
        "transcript",
        "profile",
        "fakeDevice",
        "phaseDashboard",
        "candidatePool",
        "selectedBoard",
    ]:
        assert needle in html
