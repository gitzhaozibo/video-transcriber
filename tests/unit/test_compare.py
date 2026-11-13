"""Unit tests for src.compare and scripts/compare_engines.py (Docker/APIs mocked)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import pytest

from src.compare import (
    EngineRun,
    console_summary,
    normalize_text,
    render_html,
    similarity,
    time_windows,
)
from src.subtitle import Segment

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(engine: str, segs: list[tuple[float, float, str]], **timings) -> EngineRun:
    return EngineRun(
        engine=engine,
        segments=[Segment(start=s, end=e, text=t) for s, e, t in segs],
        timings=dict(timings),
    )


class TestText:
    def test_normalize_drops_punctuation_space_and_width(self) -> None:
        assert normalize_text("Ｈｅｌｌｏ、 World！ 今日は。") == "helloworld今日は"

    def test_similarity_ignores_punctuation(self) -> None:
        assert similarity("今日は、晴れです。", "今日は晴れです") == pytest.approx(1.0)

    def test_similarity_partial(self) -> None:
        assert 0.0 < similarity("今日は晴れです", "今日は雨です") < 1.0

    def test_similarity_both_empty(self) -> None:
        assert similarity("", "。") == 1.0


class TestEngineRun:
    def test_summary(self) -> None:
        run = _run("azure", [(0, 2, "こんにちは"), (3, 7, "世界")],
                   extract=1.0, copy=0.5, transcribe=10.0)
        s = run.summary(duration=100.0, reference="こんにちは世界")
        assert s["total_seconds"] == pytest.approx(11.5)
        assert s["transcribe_rtf"] == pytest.approx(0.1)
        assert s["total_rtf"] == pytest.approx(0.115)
        assert s["segments"] == 2
        assert s["characters"] == 7
        assert s["avg_segment_seconds"] == pytest.approx(3.0)
        assert s["max_segment_seconds"] == pytest.approx(4.0)
        assert s["reference_similarity"] == pytest.approx(1.0)

    def test_failed_run_has_no_reference_score(self) -> None:
        run = EngineRun(engine="azure", error="boom")
        s = run.summary(duration=10.0, reference="x")
        assert "reference_similarity" not in s
        assert s["segments"] == 0
        assert s["total_rtf"] is None


class TestTimeWindows:
    def test_groups_by_start_and_skips_empty(self) -> None:
        runs = [
            _run("whisper", [(1, 5, "a"), (12, 15, "b"), (65, 70, "c")]),
            _run("azure", [(2, 6, "A"), (61, 64, "C")]),
        ]
        windows = time_windows(runs, 30, duration=90)
        assert windows == [
            (0, 30, {"whisper": "a b", "azure": "A"}),
            (60, 90, {"whisper": "c", "azure": "C"}),
        ]


class TestSpeakers:
    def test_speaker_labels_and_count(self) -> None:
        azure = EngineRun(engine="azure", segments=[
            Segment(0, 2, "はい", speaker=1), Segment(3, 5, "どうも", speaker=2),
            Segment(6, 8, "では", speaker=1),
        ])
        whisper = _run("whisper", [(0, 2, "はい")])
        assert azure.summary(10.0)["speakers"] == 2
        assert whisper.summary(10.0)["speakers"] is None
        (window,) = time_windows([whisper, azure], 30, duration=10)
        assert window[2] == {"whisper": "はい", "azure": "[S1] はい [S2] どうも [S1] では"}
        assert "話者数" in console_summary([whisper, azure], 10.0, None)


class TestRendering:
    def test_console_and_html(self) -> None:
        runs = [
            _run("whisper", [(0, 2, "<b>こんにちは</b>")], extract=1, model_load=5, transcribe=20),
            EngineRun(engine="azure", error="AzureTranscriptionError: HTTP 401"),
        ]
        text = console_summary(runs, 60.0, None, reference="こんにちは")
        assert "モデル読み込み" in text
        assert "HTTP 401" in text
        page = render_html(
            title="比較", source_name="a.mp4", duration=60.0, runs=runs,
            agreement=None, window_seconds=30, reference="こんにちは",
        )
        assert "&lt;b&gt;こんにちは&lt;/b&gt;" in page  # escaped
        assert "HTTP 401" in page
        assert "prefers-color-scheme: dark" in page


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "compare_engines", PROJECT_ROOT / "scripts" / "compare_engines.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # load_env is patched out by tests/conftest.py
    return mod


class TestScript:
    def test_end_to_end_with_mocks(self, tmp_path: Path) -> None:
        mod = _load_script()
        video = tmp_path / "sample.mp4"
        video.write_bytes(b"video")
        cdir = PurePosixPath("/tmp/vt-jobs/compare-x")

        def fake_copy_from(path, dest):
            dest.mkdir(parents=True, exist_ok=True)
            out = dest / PurePosixPath(path).name
            out.write_bytes(b"audio")
            return out

        def fake_whisper(audio, **kwargs):
            yield Segment(0.0, 2.0, " 今日は晴れです。")

        def fake_azure(audio_path, *, config=None):
            return [Segment(0.1, 2.1, "今日は晴れです")], {"phrases": []}

        fake_cfg = mod.azure_transcriber.AzureSpeechConfig(key="k", endpoint="https://e")
        with patch.object(mod, "container_job_dir", return_value=cdir), \
             patch.object(mod, "copy_to_container", return_value=cdir / "sample.mp4"), \
             patch.object(mod, "probe_duration", side_effect=[600.0, 60.0]), \
             patch.object(mod, "trim_media") as trim, \
             patch.object(mod, "extract_audio"), \
             patch.object(mod, "extract_speech_audio"), \
             patch.object(mod, "copy_from_container", side_effect=fake_copy_from), \
             patch.object(mod, "remove_container_path") as remove, \
             patch.object(mod.whisper_transcriber, "_get_model"), \
             patch.object(mod.whisper_transcriber, "transcribe", side_effect=fake_whisper), \
             patch.object(mod.azure_transcriber, "load_config", return_value=fake_cfg), \
             patch.object(mod.azure_transcriber, "transcribe_file", side_effect=fake_azure):
            code = mod.main([str(video), "--clip-seconds", "60", "--out", str(tmp_path / "out")])

        assert code == 0
        trim.assert_called_once()
        remove.assert_called_once_with(cdir)
        (out_dir,) = (tmp_path / "out").iterdir()
        assert {p.name for p in out_dir.iterdir()} == {
            "whisper.srt", "whisper.txt", "azure.srt", "azure.txt",
            "results.json", "report.html", "azure_response.json",
        }
        raw = json.loads((out_dir / "azure_response.json").read_text(encoding="utf-8"))
        assert raw["file"] == "speech.ogg"
        assert raw["response"] == {"phrases": []}
        results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        assert results["audio_duration_seconds"] == 60.0
        assert results["engine_agreement"] == pytest.approx(1.0)
        assert set(results["engines"]) == {"whisper", "azure"}
        assert "model_load" in results["engines"]["whisper"]["timings"]

    def test_azure_failure_still_writes_report(self, tmp_path: Path) -> None:
        mod = _load_script()
        video = tmp_path / "sample.mp4"
        video.write_bytes(b"video")
        cdir = PurePosixPath("/tmp/vt-jobs/compare-y")
        with patch.object(mod, "container_job_dir", return_value=cdir), \
             patch.object(mod, "copy_to_container", return_value=cdir / "sample.mp4"), \
             patch.object(mod, "probe_duration", return_value=30.0), \
             patch.object(mod, "remove_container_path"), \
             patch.object(mod.azure_transcriber, "load_config",
                          side_effect=mod.azure_transcriber.AzureTranscriptionConfigError("no key")):
            code = mod.main([str(video), "--engines", "azure", "--out", str(tmp_path / "out")])

        assert code == 1
        (out_dir,) = (tmp_path / "out").iterdir()
        results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        assert "no key" in results["engines"]["azure"]["error"]
        assert (out_dir / "report.html").exists()
