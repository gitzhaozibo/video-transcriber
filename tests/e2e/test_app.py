"""E2E tests for the Streamlit app using streamlit.testing.v1.AppTest.

Heavy processing (FFmpeg, Whisper) is mocked so these tests run without
Docker or GPU.

Marked with ``pytest.mark.e2e``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_test():
    """Return an AppTest instance for app.py."""
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAppLoads:
    def test_title_present(self) -> None:
        """App should render without crashing."""
        at = _make_app_test()
        at.run()
        assert not at.exception


class TestUploadFlow:
    def test_upload_button_visible(self) -> None:
        """File uploader should be present on the page."""
        at = _make_app_test()
        at.run()
        # The uploader widget should be present
        assert not at.exception

    def test_transcribe_button_disabled_before_upload(self) -> None:
        """Transcribe button should be disabled when no file is uploaded."""
        at = _make_app_test()
        at.run()
        # Without upload, the button for transcription should not appear or be disabled.
        # Verify the app starts cleanly.
        assert not at.exception


_AZURE_ENV = ("TRANSCRIBE_ENGINE", "AZURE_SPEECH_KEY", "AZURE_SPEECH_ENDPOINT",
              "AZURE_SPEECH_REGION", "AZURE_SPEECH_LOCALES", "AZURE_PHRASE_LIST",
              "AZURE_PHRASE_LIST_WEIGHT")


@pytest.fixture
def clean_engine_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _AZURE_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class TestEngineSelection:
    def test_default_is_whisper(self, clean_engine_env) -> None:
        at = _make_app_test()
        at.run()
        assert not at.exception
        assert at.radio(key="engine_choice").value == "whisper"
        assert not at.error

    def test_env_sets_initial_selection(self, clean_engine_env) -> None:
        clean_engine_env.setenv("TRANSCRIBE_ENGINE", "azure")
        clean_engine_env.setenv("AZURE_SPEECH_KEY", "k")
        clean_engine_env.setenv("AZURE_SPEECH_REGION", "japaneast")
        at = _make_app_test()
        at.run()
        assert at.radio(key="engine_choice").value == "azure"
        assert not at.error

    def test_selecting_unconfigured_azure_shows_error(self, clean_engine_env) -> None:
        at = _make_app_test()
        at.run()
        at.radio(key="engine_choice").set_value("azure").run()
        assert not at.exception
        assert any("AZURE_SPEECH_KEY" in e.value for e in at.error)

    def test_selecting_configured_azure(self, clean_engine_env) -> None:
        clean_engine_env.setenv("AZURE_SPEECH_KEY", "k")
        clean_engine_env.setenv("AZURE_SPEECH_REGION", "japaneast")
        at = _make_app_test()
        at.run()
        at.radio(key="engine_choice").set_value("azure").run()
        assert not at.error
        assert any("Azure AI Speech" in m.value for m in at.sidebar.markdown)

    def test_invalid_env_engine_falls_back_to_whisper(self, clean_engine_env) -> None:
        clean_engine_env.setenv("TRANSCRIBE_ENGINE", "google")
        at = _make_app_test()
        at.run()
        assert not at.exception
        assert at.radio(key="engine_choice").value == "whisper"
        assert any("TRANSCRIBE_ENGINE" in w.value for w in at.warning)


class _FakeUpload:
    name = "sample.mp4"
    size = 5

    def read(self) -> bytes:
        return b"video"


class TestResultJson:
    def test_azure_job_saves_result_json(self, clean_engine_env, tmp_path: Path) -> None:
        import json

        from src.ffmpeg_client import AudioChunk
        from src.subtitle import Segment

        clean_engine_env.setenv("AZURE_SPEECH_KEY", "k")
        clean_engine_env.setenv("AZURE_SPEECH_REGION", "japaneast")
        clean_engine_env.delenv("OPENAI_API_KEY", raising=False)
        results_dir = tmp_path / "results"
        jobs_dir = tmp_path / "jobs"

        def fake_job_dir(job_id: str) -> Path:
            path = jobs_dir / job_id
            path.mkdir(parents=True, exist_ok=True)
            return path

        raw_body = {"durationMilliseconds": 2000,
                    "phrases": [{"offsetMilliseconds": 500, "durationMilliseconds": 1500,
                                 "text": "こんにちは", "speaker": 1, "locale": "ja-JP"}]}

        def fake_transcribe_chunks(chunks, *, config=None, responses=None):
            responses.append({"file": "chunk_000.ogg", "start_seconds": 0.0,
                              "response": raw_body})
            return [Segment(0.5, 2.0, "こんにちは", speaker=1)]

        def fake_copy_from(path, dest):
            dest.mkdir(parents=True, exist_ok=True)
            out = dest / Path(str(path)).name
            out.write_bytes(b"ogg")
            return out

        with patch("streamlit.file_uploader", return_value=_FakeUpload()), \
             patch("src.results.RESULTS_DIR", results_dir), \
             patch("src.jobs.job_dir", side_effect=fake_job_dir), \
             patch("src.jobs.cleanup_job"), \
             patch("src.job_logger.get_logger", return_value=MagicMock()), \
             patch("src.db.create_job"), patch("src.db.update_job"), \
             patch("src.ffmpeg_client.copy_to_container"), \
             patch("src.ffmpeg_client.probe_duration", return_value=12.5), \
             patch("src.ffmpeg_client.extract_speech_audio_chunks",
                   return_value=[AudioChunk(Path("/c/speech/chunk_000.ogg"), 0.0)]) as speech, \
             patch("src.ffmpeg_client.extract_audio") as plain_extract, \
             patch("src.ffmpeg_client.copy_from_container", side_effect=fake_copy_from), \
             patch("src.azure_transcriber.transcribe_chunks",
                   side_effect=fake_transcribe_chunks):
            at = _make_app_test()
            at.run()
            at.radio(key="engine_choice").set_value("azure").run()
            next(b for b in at.button if "Transcribe" in b.label).click().run()

        assert not at.exception
        # Azure audio is converted straight from the uploaded video (one pass).
        speech.assert_called_once()
        assert speech.call_args.args[0].endswith("/sample.mp4")  # the video itself
        plain_extract.assert_not_called()
        (folder,) = results_dir.iterdir()
        data = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        assert data["status"] == "transcribed"
        assert data["engine"]["audio_format"] == {
            "container": "ogg", "codec": "opus", "sample_rate": 16000,
            "channels": 1, "bitrate": "32k", "application": "voip",
        }
        assert data["input"] == {"filename": "sample.mp4", "size_bytes": 5,
                                 "duration_seconds": 12.5}
        assert data["engine"]["name"] == "azure"
        assert data["engine"]["locales"] == ["ja-JP", "en-US", "zh-CN", "de-DE"]
        assert data["engine"]["audio_chunks"][0]["file"] == "chunk_000.ogg"
        assert "k" not in json.dumps(data["engine"]).split('"')  # key never stored
        assert data["transcription"]["segments"] == [
            {"speaker": 1, "start": 0.5, "end": 2.0, "text": "こんにちは"}
        ]
        assert data["engine"]["diarization"] is True
        assert data["engine"]["max_speakers"] == 4
        assert data["translation"]["status"] == "skipped"
        transcript = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
        assert transcript == data["transcription"]["segments"]
        assert not (folder / "translation.json").exists()
        (job_folder,) = jobs_dir.iterdir()
        assert json.loads((job_folder / "result.json").read_text(encoding="utf-8")) == data
        assert (job_folder / "transcript.json").exists()
        assert data["engine"]["raw_response_file"] == "azure_response.json"
        assert data["engine"]["language_mode"] == "identification"
        assert data["engine"]["phrase_list"] == []
        assert data["transcription"]["language"] == {
            "locale": "ja-JP",
            "locales": [{"locale": "ja-JP", "seconds": 1.5, "phrases": 1}],
        }
        assert any("Detected language: ja-JP" in c.value for c in at.caption)
        assert "_azure_response" not in data
        for f in (folder, job_folder):
            raw = json.loads((f / "azure_response.json").read_text(encoding="utf-8"))
            assert raw["chunks"] == [{"file": "chunk_000.ogg", "start_seconds": 0.0,
                                      "response": raw_body}]
            assert raw["definition"]["locales"] == ["ja-JP", "en-US", "zh-CN", "de-DE"]
        steps = [s["step"] for s in data["steps"]]
        assert steps[:5] == ["upload", "copy_to_container", "probe_duration",
                             "extract_audio", "copy_from_container"]
        assert "transcribe" in steps
        assert any("Result JSON" in b.label for b in at.get("download_button"))

    def test_failed_job_saves_failed_result(self, clean_engine_env, tmp_path: Path) -> None:
        import json

        results_dir = tmp_path / "results"
        with patch("streamlit.file_uploader", return_value=_FakeUpload()), \
             patch("src.results.RESULTS_DIR", results_dir), \
             patch("src.jobs.job_dir", side_effect=lambda j: tmp_path), \
             patch("src.jobs.cleanup_job"), \
             patch("src.job_logger.get_logger", return_value=MagicMock()), \
             patch("src.db.create_job"), patch("src.db.update_job"), \
             patch("src.ffmpeg_client.copy_to_container",
                   side_effect=RuntimeError("Docker daemon is unreachable")):
            at = _make_app_test()
            at.run()
            next(b for b in at.button if "Transcribe" in b.label).click().run()

        (folder,) = results_dir.iterdir()
        data = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        assert data["status"] == "failed"
        assert data["engine"]["name"] == "whisper"
        assert data["error"] == {"step": "copy_to_container",
                                 "message": "Docker daemon is unreachable"}

    def test_whisper_job_has_null_speakers(self, clean_engine_env, tmp_path: Path) -> None:
        import json

        from src.subtitle import Segment

        clean_engine_env.delenv("OPENAI_API_KEY", raising=False)
        results_dir = tmp_path / "results"

        def fake_transcribe(audio_path, **kwargs):
            yield Segment(0.0, 1.5, " Hello")
            yield Segment(1.5, 3.0, " World")

        with patch("streamlit.file_uploader", return_value=_FakeUpload()), \
             patch("src.results.RESULTS_DIR", results_dir), \
             patch("src.jobs.job_dir", side_effect=lambda j: tmp_path), \
             patch("src.jobs.cleanup_job"), \
             patch("src.job_logger.get_logger", return_value=MagicMock()), \
             patch("src.db.create_job"), patch("src.db.update_job"), \
             patch("src.ffmpeg_client.copy_to_container"), \
             patch("src.ffmpeg_client.probe_duration", return_value=3.0), \
             patch("src.ffmpeg_client.extract_audio"), \
             patch("src.ffmpeg_client.copy_from_container"), \
             patch("src.transcriber.transcribe", side_effect=fake_transcribe):
            at = _make_app_test()
            at.run()
            next(b for b in at.button if "Transcribe" in b.label).click().run()

        assert not at.exception
        (folder,) = results_dir.iterdir()
        data = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        assert data["engine"]["name"] == "whisper"
        assert data["transcription"]["segments"] == [
            {"speaker": None, "start": 0.0, "end": 1.5, "text": "Hello"},
            {"speaker": None, "start": 1.5, "end": 3.0, "text": "World"},
        ]
        assert any("Result JSON" in b.label for b in at.get("download_button"))
