"""Compare Whisper and Azure AI Speech transcription on the same video.

Runs the same container-side pipeline the app uses for each engine and
records per-stage timings, then writes SRT/text output, ``results.json``, a
side-by-side ``report.html`` and Azure's raw ``azure_response.json``.

Usage (from the project root, with ``docker compose up -d`` running)::

    python scripts/compare_engines.py <video> [options]

    # First 5 minutes only (Whisper large-v3 on CPU is slow on long videos)
    python scripts/compare_engines.py sample.mp4 --clip-seconds 300

    # Also score each engine against a correct transcript
    python scripts/compare_engines.py sample.mp4 --reference sample_ref.txt

Settings (Azure key, etc.) are read from ``.env`` like the app.  Azure usage is
billed per audio hour.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env import load_env  # noqa: E402

load_env()

from src import azure_transcriber  # noqa: E402
from src import transcriber as whisper_transcriber  # noqa: E402
from src.compare import (  # noqa: E402
    EngineRun,
    console_summary,
    render_html,
    similarity,
)
from src.ffmpeg_client import (  # noqa: E402
    container_job_dir,
    copy_from_container,
    copy_to_container,
    extract_audio,
    extract_speech_audio,
    probe_duration,
    remove_container_path,
    trim_media,
)
from src.subtitle import write_srt  # noqa: E402

ENGINES = ("whisper", "azure")


class _Timer:
    """Context manager recording elapsed seconds into ``run.timings[stage]``."""

    def __init__(self, run: EngineRun, stage: str) -> None:
        self.run, self.stage = run, stage

    def __enter__(self) -> None:
        print(f"  [{self.run.engine}] {self.stage} …", flush=True)
        self.started = time.perf_counter()

    def __exit__(self, *exc) -> None:
        elapsed = time.perf_counter() - self.started
        self.run.timings[self.stage] = elapsed
        status = "failed" if exc[0] else "done"
        print(f"  [{self.run.engine}] {self.stage} {status} ({elapsed:.1f}s)", flush=True)


def run_whisper(
    c_input: PurePosixPath,
    cdir: PurePosixPath,
    work: Path,
    duration: float,
    args: argparse.Namespace,
) -> EngineRun:
    run = EngineRun(
        engine="whisper",
        detail=f"faster-whisper {args.whisper_model} / {args.whisper_device} / "
        f"{args.whisper_compute_type}",
    )
    try:
        c_audio = cdir / "whisper_audio.m4a"
        with _Timer(run, "extract"):
            extract_audio(str(c_input), str(c_audio))
        with _Timer(run, "copy"):
            audio = copy_from_container(c_audio, work)
        with _Timer(run, "model_load"):
            whisper_transcriber._get_model(
                args.whisper_model, args.whisper_device, args.whisper_compute_type
            )
        with _Timer(run, "transcribe"):
            last_report = 0.0
            for seg in whisper_transcriber.transcribe(
                audio,
                model_size=args.whisper_model,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
                language=args.whisper_language,
            ):
                run.segments.append(seg)
                if seg.end - last_report >= 60 or seg.end >= duration:
                    last_report = seg.end
                    print(
                        f"    whisper {min(seg.end / duration, 1):.0%} "
                        f"({seg.end:.0f}s / {duration:.0f}s)",
                        flush=True,
                    )
    except Exception as exc:
        run.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return run


def run_azure(
    c_input: PurePosixPath,
    cdir: PurePosixPath,
    work: Path,
    args: argparse.Namespace,
) -> EngineRun:
    run = EngineRun(engine="azure")
    try:
        config = azure_transcriber.load_config()
        run.detail = f"fast transcription / {config.language_label}"
        c_speech = cdir / "speech.ogg"
        with _Timer(run, "extract"):
            extract_speech_audio(str(c_input), str(c_speech))
        with _Timer(run, "copy"):
            speech_path = copy_from_container(c_speech, work)
        run.detail += f" / {speech_path.stat().st_size / 1_000_000:.1f} MB"
        with _Timer(run, "transcribe"):
            run.segments, response = azure_transcriber.transcribe_file(
                speech_path, config=config
            )
        run.raw_response = azure_transcriber.response_document(
            config, speech_path.name, response
        )
    except Exception as exc:
        run.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Whisper and Azure AI Speech transcription on one video."
    )
    parser.add_argument("video", type=Path, help="Input video (or audio) file")
    parser.add_argument(
        "--engines",
        default="whisper,azure",
        help="Comma-separated engines to run (default: whisper,azure)",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Only use the first N seconds of the input",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Correct transcript (UTF-8 text) to score each engine against",
    )
    parser.add_argument("--whisper-model", default=whisper_transcriber.DEFAULT_MODEL)
    parser.add_argument("--whisper-device", default=whisper_transcriber.DEFAULT_DEVICE)
    parser.add_argument(
        "--whisper-compute-type", default=whisper_transcriber.DEFAULT_COMPUTE_TYPE
    )
    parser.add_argument(
        "--whisper-language",
        default=None,
        help="Whisper language code such as ja (default: auto-detect, like the app)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=30.0,
        help="Time window for the side-by-side view in report.html (default: 30)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "comparison",
        help="Output root directory (default: comparison/)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the extracted audio files (host and container)",
    )
    args = parser.parse_args(argv)
    args.engines = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in args.engines if e not in ENGINES]
    if unknown or not args.engines:
        parser.error(f"--engines must be a subset of {','.join(ENGINES)}: {unknown}")
    if not args.video.is_file():
        parser.error(f"Input file not found: {args.video}")
    if args.reference is not None and not args.reference.is_file():
        parser.error(f"Reference file not found: {args.reference}")
    return args


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:  # don't crash on characters the console code page can't show
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out / f"{stamp}_{args.video.stem}"
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    cdir = container_job_dir(f"compare-{stamp}")
    reference = (
        args.reference.read_text(encoding="utf-8") if args.reference is not None else None
    )

    print(f"Output: {out_dir}")
    runs: list[EngineRun] = []
    try:
        print("Copying input to the FFmpeg container …", flush=True)
        c_input = copy_to_container(args.video, cdir)
        duration = probe_duration(str(c_input))
        if args.clip_seconds and args.clip_seconds < duration:
            c_clip = cdir / f"clip{args.video.suffix}"
            trim_media(str(c_input), str(c_clip), args.clip_seconds)
            c_input = c_clip
            duration = probe_duration(str(c_input))
        print(f"Audio duration: {duration:.1f}s", flush=True)

        for engine in args.engines:
            print(f"\n=== {engine} ===", flush=True)
            if engine == "whisper":
                runs.append(run_whisper(c_input, cdir, work, duration, args))
            else:
                runs.append(run_azure(c_input, cdir, work, args))
    finally:
        if not args.keep_audio:
            try:
                remove_container_path(cdir)
            except Exception as exc:  # pragma: no cover - best effort
                print(f"warning: could not clean container dir {cdir}: {exc}")
            shutil.rmtree(work, ignore_errors=True)

    ok_runs = [r for r in runs if r.ok]
    agreement = (
        round(similarity(ok_runs[0].text, ok_runs[1].text), 4) if len(ok_runs) == 2 else None
    )

    for run in runs:
        if run.ok:
            write_srt(run.segments, out_dir / f"{run.engine}.srt")
            (out_dir / f"{run.engine}.txt").write_text(run.text + "\n", encoding="utf-8")
        if run.raw_response is not None:
            (out_dir / f"{run.engine}_response.json").write_text(
                json.dumps(run.raw_response, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    results = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": args.video.name,
        "clip_seconds": args.clip_seconds,
        "audio_duration_seconds": round(duration, 3),
        "engine_agreement": agreement,
        "reference": args.reference.name if args.reference else None,
        "engines": {r.engine: r.summary(duration, reference) for r in runs},
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    notes = []
    if args.clip_seconds:
        notes.append(f"入力の先頭 {args.clip_seconds:g} 秒のみで比較")
    if "whisper" in args.engines:
        notes.append("Whisper のモデル読み込み時間は初回のモデルダウンロードを含む場合があります")
    if "azure" in args.engines:
        notes.append("Azure の文字起こし時間はアップロードとネットワーク往復を含みます")
    (out_dir / "report.html").write_text(
        render_html(
            title="文字起こしエンジン比較",
            source_name=args.video.name,
            duration=duration,
            runs=runs,
            agreement=agreement,
            window_seconds=args.window_seconds,
            reference=reference,
            notes=notes,
        ),
        encoding="utf-8",
    )

    print()
    print(console_summary(runs, duration, agreement, reference))
    print(f"\nReport: {out_dir / 'report.html'}")
    return 0 if all(r.ok for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
