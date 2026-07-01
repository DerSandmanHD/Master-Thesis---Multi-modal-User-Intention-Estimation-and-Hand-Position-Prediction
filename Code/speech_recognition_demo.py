#!/usr/bin/env python3
"""Detect spoken phase commands using speech-event timing and windowed Whisper classification."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain


EXPECTED_COMMAND_ORDER = ("START", "SECOND", "DONE", "THIRD")
COMMAND_ALIASES = {
    "START": ("start", "started", "star", "stark", "stat"),
    "SECOND": ("second", "seconds", "secondly", "sicken", "secon"),
    "DONE": ("done", "down", "dan", "dawn"),
    "THIRD": ("third", "thirst", "turd", "the third"),
}
FUZZY_THRESHOLD = 0.78
WHISPER_SAMPLE_RATE = 16_000
VAD_TIMESTAMP_UNCERTAINTY_MS = 32


def _rms_normalize(audio: np.ndarray, target_dbfs: float = -18.0) -> np.ndarray:
    """Raise quiet speech without letting short peaks dominate the gain."""
    if audio.size == 0:
        return audio

    audio = audio.astype(np.float32)
    audio = audio - float(np.mean(audio))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms <= 1e-8:
        return audio

    target_rms = 10 ** (target_dbfs / 20.0)
    audio = audio * min(target_rms / rms, 50.0)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio = audio / peak * 0.98
    return audio


def _write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.clip(samples, -0.98, 0.98)
    pcm_int16 = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_int16.tobytes())


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, found sample width {sample_width}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def wav_duration_seconds(path: str | Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def extract_audio_gen2(vrs_file_path: str, provider=None) -> str | None:
    """Extract duration-preserving mono audio from an Aria Gen 2 VRS file."""
    if provider is None:
        provider = data_provider.create_vrs_data_provider(vrs_file_path)

    stream_id = provider.get_stream_id_from_label("mic")
    config = provider.get_audio_configuration(stream_id)
    sample_rate = config.sample_rate
    num_channels = config.num_channels
    num_blocks = provider.get_num_data(stream_id)
    if num_blocks == 0:
        return None

    all_samples = []
    for index in range(num_blocks):
        audio_data, record = provider.get_audio_data_by_index(stream_id, index)
        block = np.array(audio_data.data, dtype=np.float32)
        if num_channels > 1:
            block = block.reshape(-1, num_channels).mean(axis=1)
        # Muted blocks must remain in the timeline; dropping them shifts all later timestamps.
        if record.audio_muted:
            block = np.zeros_like(block)
        all_samples.append(block)

    if not all_samples:
        return None

    pcm_float = _rms_normalize(np.concatenate(all_samples))
    temp_folder = tempfile.mkdtemp(prefix="aria_command_audio_")
    wav_path = os.path.join(temp_folder, "audio_raw.wav")
    _write_wav(wav_path, pcm_float, sample_rate)
    return wav_path


def preprocess_audio_for_whisper(raw_wav_path: str) -> str:
    """Create a 16-kHz Whisper input while preserving the original duration."""
    normalized_path = os.path.join(os.path.dirname(raw_wav_path), "audio_whisper.wav")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        raw_wav_path,
        "-vn",
        "-ar",
        str(WHISPER_SAMPLE_RATE),
        "-ac",
        "1",
        "-af",
        "highpass=f=80,acompressor=threshold=-30dB:ratio=6:attack=5:release=80:makeup=10,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a",
        "pcm_s16le",
        normalized_path,
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return raw_wav_path

    duration_delta = abs(wav_duration_seconds(normalized_path) - wav_duration_seconds(raw_wav_path))
    if duration_delta > 0.02:
        return raw_wav_path
    return normalized_path


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def command_scores(text: str) -> dict[str, float]:
    """Return lexical command scores without treating them as timing confidence."""
    normalized_text = normalize_text(text)
    if not normalized_text:
        return {command: 0.0 for command in EXPECTED_COMMAND_ORDER}

    tokens = normalized_text.split()
    scores = {}
    for command, aliases in COMMAND_ALIASES.items():
        best_score = 0.0
        for alias in aliases:
            alias_tokens = alias.split()
            width = len(alias_tokens)
            ngrams = [
                " ".join(tokens[index:index + width])
                for index in range(max(1, len(tokens) - width + 1))
            ]
            if alias in ngrams:
                best_score = 1.0
                break
            for ngram in ngrams:
                best_score = max(best_score, SequenceMatcher(None, ngram, alias).ratio())
        scores[command] = best_score
    return scores


def match_command_text(text: str) -> tuple[str | None, float, list[str]]:
    scores = command_scores(text)
    plausible = [command for command, score in scores.items() if score >= FUZZY_THRESHOLD]
    if not plausible:
        return None, max(scores.values(), default=0.0), []

    plausible.sort(key=lambda command: scores[command], reverse=True)
    best = plausible[0]
    ambiguous = [command for command in plausible[1:] if scores[best] - scores[command] < 0.1]
    if ambiguous:
        return None, scores[best], [best, *ambiguous]
    return best, scores[best], plausible


def energy_speech_windows(audio: np.ndarray, sample_rate: int, args) -> list[dict]:
    """Fallback speech segmentation when the bundled Silero VAD is unavailable."""
    from scipy import ndimage, signal

    nyquist = sample_rate / 2.0
    high_cutoff = min(4_000.0, nyquist * 0.95)
    filtered = signal.sosfiltfilt(
        signal.butter(4, [80.0 / nyquist, high_cutoff / nyquist], btype="bandpass", output="sos"),
        audio,
    )
    frame_length = max(1, int(sample_rate * 0.02))
    hop_length = max(1, int(sample_rate * 0.01))
    frame_count = max(1, 1 + (len(filtered) - frame_length) // hop_length)
    rms = np.empty(frame_count, dtype=np.float64)
    for index in range(frame_count):
        frame = filtered[index * hop_length:index * hop_length + frame_length]
        rms[index] = np.sqrt(np.mean(frame * frame) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    noise_floor = float(np.percentile(db, 20))
    threshold = min(noise_floor + 10.0, float(np.percentile(db, 90)) - 5.0)
    active = db >= threshold

    close_frames = max(1, int(args.min_silence_ms / 10))
    active = ndimage.binary_closing(active, structure=np.ones(close_frames, dtype=bool))
    labels, label_count = ndimage.label(active)
    windows = []
    minimum_frames = max(1, int(args.min_speech_ms / 10))
    for label in range(1, label_count + 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) < minimum_frames:
            continue
        start_sample = int(indices[0] * hop_length)
        end_sample = int(min(len(audio), indices[-1] * hop_length + frame_length))
        windows.append({"start": start_sample, "end": end_sample})
    return windows


def detect_speech_windows(audio: np.ndarray, sample_rate: int, args) -> tuple[list[dict], str]:
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        options = VadOptions(
            threshold=args.vad_threshold,
            neg_threshold=args.vad_neg_threshold,
            min_speech_duration_ms=args.min_speech_ms,
            max_speech_duration_s=args.max_speech_seconds,
            min_silence_duration_ms=args.min_silence_ms,
            speech_pad_ms=0,
        )
        return get_speech_timestamps(audio, options, sampling_rate=sample_rate), "silero_vad"
    except Exception:
        return energy_speech_windows(audio, sample_rate, args), "energy_fallback"


def transcribe_speech_window(model, audio: np.ndarray, sample_rate: int, window: dict, args) -> dict:
    context_samples = int(args.window_context_seconds * sample_rate)
    clip_start = max(0, int(window["start"]) - context_samples)
    clip_end = min(len(audio), int(window["end"]) + context_samples)
    clip = audio[clip_start:clip_end]

    segments, _ = model.transcribe(
        clip,
        language="en",
        word_timestamps=False,
        vad_filter=False,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        hotwords="start second done third",
    )
    segments = list(segments)
    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    if segments:
        avg_logprob = float(np.mean([segment.avg_logprob for segment in segments]))
        no_speech_prob = float(max(segment.no_speech_prob for segment in segments))
    else:
        avg_logprob = None
        no_speech_prob = 1.0

    command, lexical_score, plausible_commands = match_command_text(transcript)
    warnings = []
    if len(plausible_commands) > 1:
        warnings.append("multiple_commands_in_speech_window")
    if command is not None and avg_logprob is not None and avg_logprob < args.min_avg_logprob:
        warnings.append("low_acoustic_logprob")
    if command is not None and no_speech_prob > args.max_no_speech_prob:
        warnings.append("high_no_speech_probability")

    start_seconds = int(window["start"]) / sample_rate
    end_seconds = int(window["end"]) / sample_rate
    return {
        "speech_start_seconds": round(start_seconds, 3),
        "speech_end_seconds": round(end_seconds, 3),
        "speech_duration_seconds": round(end_seconds - start_seconds, 3),
        "classification_window_start_seconds": round(clip_start / sample_rate, 3),
        "classification_window_end_seconds": round(clip_end / sample_rate, 3),
        "transcript": transcript,
        "command": command,
        "match_score": round(lexical_score, 3),
        "plausible_commands": plausible_commands,
        "avg_logprob": round(avg_logprob, 4) if avg_logprob is not None else None,
        "no_speech_probability": round(no_speech_prob, 4),
        "warnings": warnings,
    }


def classify_speech_windows(model, audio: np.ndarray, sample_rate: int, windows: list[dict], args) -> tuple[list[dict], list[dict]]:
    events = []
    candidates = []
    for event_id, window in enumerate(windows):
        event = transcribe_speech_window(model, audio, sample_rate, window, args)
        event["event_id"] = event_id
        events.append(event)
        if event["command"] is None:
            continue
        candidates.append(
            {
                "event_id": event_id,
                "command": event["command"],
                "raw_word": event["transcript"],
                "segment_text": event["transcript"],
                "match_score": event["match_score"],
                "avg_logprob": event["avg_logprob"],
                "no_speech_probability": event["no_speech_probability"],
                "relative_seconds": event["speech_start_seconds"],
                "speech_start_seconds": event["speech_start_seconds"],
                "speech_end_seconds": event["speech_end_seconds"],
                "timestamp_source": "speech_window_start",
                "timestamp_uncertainty_ms": VAD_TIMESTAMP_UNCERTAINTY_MS,
                "warnings": event["warnings"],
            }
        )
    return events, candidates


def candidate_quality(candidate: dict) -> tuple[float, float, float]:
    avg_logprob = candidate["avg_logprob"] if candidate["avg_logprob"] is not None else -10.0
    return candidate["match_score"], avg_logprob, -candidate["relative_seconds"]


def resolve_command_sequence(candidates: list[dict], min_gap_seconds: float = 0.4) -> tuple[dict, list[str]]:
    """Pick distinct monotonic command events while allowing genuinely missing commands."""
    result = {}
    warnings = []
    search_start = -1.0
    used_events = set()

    for command in EXPECTED_COMMAND_ORDER:
        possible = [
            candidate
            for candidate in candidates
            if candidate["command"] == command
            and candidate["event_id"] not in used_events
            and candidate["relative_seconds"] >= search_start + min_gap_seconds
        ]
        if not possible:
            warnings.append(f"missing_{command.lower()}")
            continue

        selected = max(possible, key=candidate_quality)
        result[command] = selected
        used_events.add(selected["event_id"])
        search_start = selected["relative_seconds"]
        warnings.extend(selected.get("warnings", []))

    phase_pairs = (("START", "SECOND", "continue"), ("SECOND", "DONE", "fetch"), ("DONE", "THIRD", "handover"))
    for start_command, end_command, phase_name in phase_pairs:
        if start_command in result and end_command in result:
            duration = result[end_command]["relative_seconds"] - result[start_command]["relative_seconds"]
            if duration < min_gap_seconds:
                warnings.append(f"implausibly_short_{phase_name}_phase")

    warnings = list(dict.fromkeys(warnings))
    return result, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect START/SECOND/DONE/THIRD from isolated speech events rather than full-track word timestamps."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dir", type=Path, help="Directory containing .vrs files.")
    source.add_argument("--file", type=Path, help="Process one .vrs file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for summary/debug outputs. Defaults to the input directory for --dir "
            "and a separate audio_single_results directory for --file."
        ),
    )
    parser.add_argument("--pattern", default="*.vrs", help="Input glob when --dir is used. Default: *.vrs")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N sorted files.")
    parser.add_argument("--model-size", default="medium.en", help="Faster-Whisper model or local model path.")
    parser.add_argument("--device", default="cpu", help="Faster-Whisper device: cpu or cuda.")
    parser.add_argument("--compute-type", default="int8", help="For example int8 on CPU or float16 on CUDA.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download a missing Whisper model.")
    parser.add_argument("--vad-threshold", type=float, default=0.15, help="Sensitive Silero speech threshold.")
    parser.add_argument("--vad-neg-threshold", type=float, default=0.05, help="Silero speech exit threshold.")
    parser.add_argument("--min-speech-ms", type=int, default=80)
    parser.add_argument("--min-silence-ms", type=int, default=250)
    parser.add_argument("--max-speech-seconds", type=float, default=5.0)
    parser.add_argument("--window-context-seconds", type=float, default=0.3)
    parser.add_argument("--min-command-gap-seconds", type=float, default=0.4)
    parser.add_argument("--min-avg-logprob", type=float, default=-1.2)
    parser.add_argument("--max-no-speech-prob", type=float, default=0.5)
    parser.add_argument(
        "--manual-overrides",
        type=Path,
        default=None,
        help="Optional persistent JSON with manually verified command times. Defaults to timestamps_manual_overrides.json when present.",
    )
    parser.add_argument("--ignore-manual-overrides", action="store_true")
    parser.add_argument("--keep-debug-audio", action="store_true")
    parser.add_argument("--vad", action="store_true", help="Deprecated compatibility flag; event VAD is always enabled.")
    return parser.parse_args()


def discover_vrs_files(args) -> tuple[list[str], Path]:
    if args.file is not None:
        file_path = args.file.expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"VRS file not found: {file_path}")
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else file_path.parent / "audio_single_results" / file_path.stem
        )
        return [str(file_path)], output_dir

    directory = args.dir.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Input directory not found: {directory}")
    files = sorted(glob.glob(str(directory / args.pattern)))
    if args.limit is not None:
        files = files[:max(args.limit, 0)]
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else directory
    return files, output_dir


def backup_existing_output(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.before_speech_window_fix_{timestamp}{path.suffix}")
    shutil.copy2(path, backup)


def atomic_json_dump(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(data, handle, indent=2, ensure_ascii=False)
    temp_path.replace(path)
    path.chmod(0o644)


def load_manual_overrides(args, output_dir: Path) -> tuple[dict, Path | None]:
    if args.ignore_manual_overrides:
        return {}, None
    override_path = args.manual_overrides or output_dir / "timestamps_manual_overrides.json"
    override_path = override_path.expanduser().resolve()
    if not override_path.exists():
        return {}, override_path
    with override_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Manual overrides must contain a JSON object: {override_path}")
    return data, override_path


def parse_manual_time(value, audio_start_timestamp_ns: int) -> tuple[float, int]:
    if isinstance(value, (int, float)):
        relative_seconds = float(value)
        timestamp_ns = int(audio_start_timestamp_ns + relative_seconds * 1e9)
        return relative_seconds, timestamp_ns
    if not isinstance(value, dict):
        raise ValueError("Manual command time must be a number or object")

    relative_value = value.get("relative_seconds")
    timestamp_value = value.get("timestamp_ns")
    if relative_value is None and timestamp_value is None:
        raise ValueError("Manual command time requires relative_seconds or timestamp_ns")
    if relative_value is None:
        timestamp_ns = int(timestamp_value)
        relative_seconds = (timestamp_ns - audio_start_timestamp_ns) / 1e9
    else:
        relative_seconds = float(relative_value)
        timestamp_ns = int(timestamp_value) if timestamp_value is not None else int(
            audio_start_timestamp_ns + relative_seconds * 1e9
        )
    if relative_seconds < 0.0:
        raise ValueError("Manual relative_seconds must be non-negative")
    return relative_seconds, timestamp_ns


def apply_manual_overrides(
    summary_data: dict,
    debug_data: dict,
    overrides: dict,
    min_gap_seconds: float,
) -> None:
    for filename, command_overrides in overrides.items():
        if filename not in summary_data or not isinstance(command_overrides, dict):
            continue
        debug = debug_data[filename]
        audio_start_timestamp_ns = debug.get("audio_start_timestamp_ns")
        if audio_start_timestamp_ns is None:
            continue

        applied = []
        for command_name, value in command_overrides.items():
            command = command_name.upper()
            if command not in EXPECTED_COMMAND_ORDER:
                debug.setdefault("warnings", []).append(f"invalid_manual_command:{command_name}")
                continue
            try:
                relative_seconds, timestamp_ns = parse_manual_time(value, int(audio_start_timestamp_ns))
            except (TypeError, ValueError) as exc:
                debug.setdefault("warnings", []).append(f"invalid_manual_override_{command.lower()}:{exc}")
                continue

            summary_data[filename][command] = {
                "timestamp_ns": timestamp_ns,
                "relative_seconds": round(relative_seconds, 3),
                "raw_word": "manual_override",
                "match_score": 1.0,
                "avg_logprob": None,
                "speech_end_seconds": None,
                "timestamp_source": "manual_override",
                "timestamp_uncertainty_ms": None,
            }
            applied.append(command)
            missing_warning = f"missing_{command.lower()}"
            debug["warnings"] = [warning for warning in debug.get("warnings", []) if warning != missing_warning]

        if applied:
            debug["manual_overrides_applied"] = applied
        commands = summary_data[filename]
        available_order = [command for command in EXPECTED_COMMAND_ORDER if command in commands]
        available_times = [commands[command]["relative_seconds"] for command in available_order]
        if available_times != sorted(available_times) or len(set(available_times)) != len(available_times):
            debug.setdefault("warnings", []).append("manual_override_non_monotonic")

        phase_pairs = (
            ("START", "SECOND", "continue"),
            ("SECOND", "DONE", "fetch"),
            ("DONE", "THIRD", "handover"),
        )
        for start_command, end_command, phase_name in phase_pairs:
            if start_command in commands and end_command in commands:
                duration = (
                    commands[end_command]["relative_seconds"]
                    - commands[start_command]["relative_seconds"]
                )
                if duration < min_gap_seconds:
                    debug.setdefault("warnings", []).append(
                        f"implausibly_short_{phase_name}_phase"
                    )
        debug["warnings"] = list(dict.fromkeys(debug.get("warnings", [])))

        if len(commands) == len(EXPECTED_COMMAND_ORDER) and not debug["warnings"]:
            debug["quality"] = "manual_override_complete"
            debug["review_required"] = False
        else:
            debug["quality"] = "manual_review"
            debug["review_required"] = True


def process_vrs_file(vrs_path: str, model, args, debug_dir: Path | None) -> tuple[dict, dict]:
    filename = os.path.basename(vrs_path)
    provider = data_provider.create_vrs_data_provider(vrs_path)
    if provider is None:
        raise RuntimeError("could_not_open_vrs")

    audio_stream_id = provider.get_stream_id_from_label("mic")
    audio_starting_timestamp = provider.get_first_time_ns(audio_stream_id, TimeDomain.DEVICE_TIME)
    raw_audio_path = extract_audio_gen2(vrs_path, provider=provider)
    if not raw_audio_path:
        raise RuntimeError("no_audio")

    try:
        whisper_audio_path = preprocess_audio_for_whisper(raw_audio_path)
        audio, sample_rate = read_wav_mono(whisper_audio_path)
        windows, timing_method = detect_speech_windows(audio, sample_rate, args)
        events, candidates = classify_speech_windows(model, audio, sample_rate, windows, args)
        resolved_commands, warnings = resolve_command_sequence(candidates, args.min_command_gap_seconds)

        for candidate in candidates:
            candidate["timestamp_ns"] = int(
                audio_starting_timestamp + candidate["relative_seconds"] * 1e9
            )

        summary = {}
        for command, info in resolved_commands.items():
            summary[command] = {
                "timestamp_ns": info["timestamp_ns"],
                "relative_seconds": info["relative_seconds"],
                "raw_word": info["raw_word"],
                "match_score": info["match_score"],
                "avg_logprob": info["avg_logprob"],
                "speech_end_seconds": info["speech_end_seconds"],
                "timestamp_source": info["timestamp_source"],
                "timestamp_uncertainty_ms": info["timestamp_uncertainty_ms"],
            }

        quality = "auto_accept" if len(summary) == len(EXPECTED_COMMAND_ORDER) and not warnings else "manual_review"
        debug = {
            "algorithm_version": "speech_windows_v1",
            "whisper_model": args.model_size,
            "whisper_device": args.device,
            "whisper_compute_type": args.compute_type,
            "timing_method": timing_method,
            "timestamp_definition": "speech_event_start",
            "audio_start_timestamp_ns": int(audio_starting_timestamp),
            "audio_duration_seconds": round(len(audio) / sample_rate, 3),
            "vad_parameters": {
                "threshold": args.vad_threshold,
                "neg_threshold": args.vad_neg_threshold,
                "min_speech_ms": args.min_speech_ms,
                "min_silence_ms": args.min_silence_ms,
                "max_speech_seconds": args.max_speech_seconds,
            },
            "quality": quality,
            "review_required": quality != "auto_accept",
            "warnings": warnings,
            "speech_events": events,
            "candidates": candidates,
        }

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(whisper_audio_path, debug_dir / f"{Path(filename).stem}.wav")
        return summary, debug
    finally:
        shutil.rmtree(os.path.dirname(raw_audio_path), ignore_errors=True)


def main() -> int:
    args = parse_args()
    try:
        vrs_files, output_dir = discover_vrs_files(args)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}")
        return 2
    if not vrs_files:
        print("No VRS files found.")
        return 0

    print(
        f"Loading Whisper ({args.model_size}, device={args.device}, "
        f"compute_type={args.compute_type})..."
    )
    model = WhisperModel(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        local_files_only=args.local_files_only,
    )

    summary_data = {}
    debug_data = {}
    debug_dir = output_dir / "debug_audio" if args.keep_debug_audio else None

    print(f"Processing {len(vrs_files)} file(s)...")
    for vrs_path in vrs_files:
        filename = os.path.basename(vrs_path)
        print(f"  {filename}: ", end="", flush=True)
        try:
            summary, debug = process_vrs_file(vrs_path, model, args, debug_dir)
            summary_data[filename] = summary
            debug_data[filename] = debug
            print(f"{debug['quality']} ({', '.join(debug['warnings']) or 'no warnings'})")
        except Exception as exc:
            summary_data[filename] = {}
            debug_data[filename] = {
                "quality": "error",
                "review_required": True,
                "warnings": [f"processing_error:{type(exc).__name__}"],
                "error": str(exc),
            }
            print(f"error: {exc}")

    try:
        manual_overrides, manual_override_path = load_manual_overrides(args, output_dir)
        apply_manual_overrides(
            summary_data,
            debug_data,
            manual_overrides,
            args.min_command_gap_seconds,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error loading manual overrides: {exc}")
        return 2

    print("\nResolved command timestamps")
    for filename, commands in summary_data.items():
        print(f"\n{filename}")
        for command, info in sorted(commands.items(), key=lambda item: item[1]["timestamp_ns"]):
            print(
                f"  [{info['relative_seconds']:>7.3f}s] {command:<7} "
                f"source={info['timestamp_source']} text={info['raw_word']!r}"
            )
        for warning in debug_data.get(filename, {}).get("warnings", []):
            print(f"  WARNING: {warning}")

    summary_path = output_dir / "timestamps_summary.json"
    debug_path = output_dir / "timestamps_debug.json"
    review_path = output_dir / "timestamps_review_queue.json"
    review_queue = {
        filename: {
            "quality": debug.get("quality"),
            "warnings": debug.get("warnings", []),
            "resolved_commands": summary_data.get(filename, {}),
            "speech_events": debug.get("speech_events", []),
        }
        for filename, debug in debug_data.items()
        if debug.get("review_required", True)
    }
    backup_existing_output(summary_path)
    backup_existing_output(debug_path)
    backup_existing_output(review_path)
    atomic_json_dump(summary_data, summary_path)
    atomic_json_dump(debug_data, debug_path)
    atomic_json_dump(review_queue, review_path)
    auto_accept_count = sum(debug.get("quality") == "auto_accept" for debug in debug_data.values())
    accepted_count = sum(not debug.get("review_required", True) for debug in debug_data.values())
    print(f"\nSummary: {summary_path}")
    print(f"Debug:   {debug_path}")
    print(f"Review:  {review_path}")
    if manual_override_path is not None:
        print(f"Overrides: {manual_override_path}")
    print(f"Auto-accepted: {auto_accept_count}/{len(debug_data)}")
    print(f"Accepted without review: {accepted_count}/{len(debug_data)}")
    print(f"Manual review: {len(review_queue)}/{len(debug_data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
