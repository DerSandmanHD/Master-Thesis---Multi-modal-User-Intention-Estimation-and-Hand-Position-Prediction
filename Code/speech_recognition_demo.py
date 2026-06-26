import argparse
import os
import glob
import shutil
import subprocess
import tempfile
import wave
import json
from difflib import SequenceMatcher
import numpy as np

from faster_whisper import WhisperModel
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain

EXPECTED_COMMAND_ORDER = ("START", "SECOND", "DONE", "THIRD")
COMMAND_ALIASES = {
    "START": ("start", "started", "star", "stark", "stat"),
    "SECOND": ("second", "seconds", "secondly", "sicken", "secon"),
    "DONE": ("done", "down", "dan", "dawn"),
    "THIRD": ("third", "thirst", "turd", "heard", "the third"),
}
FUZZY_THRESHOLD = 0.78


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
    gain = min(target_rms / rms, 50.0)
    audio = audio * gain

    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio = audio / peak * 0.98

    return audio


def _write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.clip(samples, -0.98, 0.98)
    pcm_int16 = (samples * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())


def extract_audio_gen2(vrs_file_path: str, provider=None) -> str | None:
    """Extract audio from Aria Gen 2 VRS (non-PCM) via DataProvider."""
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
    for i in range(num_blocks):
        audio_data, record = provider.get_audio_data_by_index(stream_id, i)
        if record.audio_muted:
            continue
        block = np.array(audio_data.data, dtype=np.float32)
        if num_channels > 1:
            block = block.reshape(-1, num_channels).mean(axis=1)
        all_samples.append(block)

    if not all_samples:
        return None

    pcm_float = np.concatenate(all_samples)
    pcm_float = _rms_normalize(pcm_float)

    temp_folder = tempfile.mkdtemp()
    wav_path = os.path.join(temp_folder, "audio_raw.wav")
    _write_wav(wav_path, pcm_float, sample_rate)

    return wav_path


def preprocess_audio_for_whisper(raw_wav_path: str) -> str:
    """Normalize speech for Whisper. Uses ffmpeg when available, otherwise keeps RMS-normalized WAV."""
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
        "16000",
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
        return normalized_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return raw_wav_path


def match_command(word: str) -> tuple[str | None, float]:
    clean_word = word.lower().strip(" .,!?;:\"'()[]{}")
    if not clean_word:
        return None, 0.0

    best_command = None
    best_score = 0.0
    for command, aliases in COMMAND_ALIASES.items():
        for alias in aliases:
            if alias in clean_word:
                score = 1.0
            else:
                score = SequenceMatcher(None, clean_word, alias).ratio()
            if score > best_score:
                best_command = command
                best_score = score

    if best_score >= FUZZY_THRESHOLD:
        return best_command, best_score
    return None, best_score


def resolve_command_sequence(candidates: list[dict]) -> tuple[dict, list[str]]:
    """Pick one monotonic START->SECOND->DONE->THIRD sequence from noisy word candidates."""
    result = {}
    warnings = []
    search_start = -1.0

    for command in EXPECTED_COMMAND_ORDER:
        possible = [
            candidate
            for candidate in candidates
            if candidate["command"] == command and candidate["relative_seconds"] > search_start
        ]
        if not possible:
            warnings.append(f"missing_{command.lower()}")
            continue

        # Prefer high confidence, but keep the expected chronological order.
        selected = max(possible, key=lambda item: (item["match_score"], -item["relative_seconds"]))
        result[command] = selected
        search_start = selected["relative_seconds"]

    if len(result) == len(EXPECTED_COMMAND_ORDER):
        times = [result[command]["relative_seconds"] for command in EXPECTED_COMMAND_ORDER]
        if times != sorted(times):
            warnings.append("non_monotonic_sequence")

    return result, warnings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the directory containing the .vrs files.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="medium.en",
        help="Faster-Whisper model size. For English trigger words, medium.en is a good default.",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable VAD. Leave disabled for very quiet short commands unless testing proves it helps.",
    )
    parser.add_argument(
        "--keep-debug-audio",
        action="store_true",
        help="Keep normalized WAV files next to timestamps_summary.json for manual inspection.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Alle VRS Dateien im Ordner finden
    vrs_files = glob.glob(os.path.join(args.dir, "*.vrs"))
    if not vrs_files:
        print(f"Keine .vrs Dateien im Ordner '{args.dir}' gefunden.")
        return

    # 2. Whisper Model laden
    print(f"Lade Whisper Model ({args.model_size})...")
    model = WhisperModel(args.model_size, device="cpu", compute_type="int8")
    
    # Hier speichern wir alle Ergebnisse
    summary_data = {}
    debug_data = {}
    debug_dir = os.path.join(args.dir, "debug_audio")
    if args.keep_debug_audio:
        os.makedirs(debug_dir, exist_ok=True)

    print(f"\nStarte Batch-Verarbeitung für {len(vrs_files)} Dateien...")
    print("-" * 50)

    # 3. Schleife über alle Videos
    for vrs_path in sorted(vrs_files):
        filename = os.path.basename(vrs_path)
        print(f"Verarbeite: {filename} ... ", end="", flush=True)
        
        summary_data[filename] = {} # Initialisiere leeres dict für dieses Video
        
        raw_audio_path = extract_audio_gen2(vrs_path)
        if not raw_audio_path:
            print("Fehler (Kein Audio)")
            continue

        try:
            whisper_audio_path = preprocess_audio_for_whisper(raw_audio_path)
            provider = data_provider.create_vrs_data_provider(vrs_path)
            audio_stream_id = provider.get_stream_id_from_label("mic")
            audio_starting_timestamp = provider.get_first_time_ns(audio_stream_id, TimeDomain.DEVICE_TIME)
            
            segments, _ = model.transcribe(
                whisper_audio_path,
                language="en",
                word_timestamps=True,
                vad_filter=args.vad,
                beam_size=5,
                condition_on_previous_text=False,
            )
            
            s_to_ns = int(1e9)
            candidates = []
            
            for segment in segments:
                if segment.words is None:
                    continue
                for word_info in segment.words:
                    command, score = match_command(word_info.word)
                    
                    if command is not None:
                        candidates.append({
                            "command": command,
                            "raw_word": word_info.word.strip(),
                            "match_score": round(score, 3),
                            "timestamp_ns": int(word_info.start * s_to_ns + audio_starting_timestamp),
                            "relative_seconds": round(word_info.start, 2),
                            "segment_text": segment.text.strip(),
                        })

            resolved_commands, warnings = resolve_command_sequence(candidates)

            for command, info in resolved_commands.items():
                summary_data[filename][command] = {
                    "timestamp_ns": info["timestamp_ns"],
                    "relative_seconds": info["relative_seconds"],
                    "raw_word": info["raw_word"],
                    "match_score": info["match_score"],
                }

            if warnings:
                debug_data.setdefault(filename, {})["warnings"] = warnings
            if candidates:
                debug_data.setdefault(filename, {})["candidates"] = candidates

            if args.keep_debug_audio:
                debug_audio_path = os.path.join(debug_dir, f"{os.path.splitext(filename)[0]}.wav")
                shutil.copyfile(whisper_audio_path, debug_audio_path)
            
            status = "Fertig"
            if warnings:
                status += f" (Warnungen: {', '.join(warnings)})"
            print(status)

        except Exception as e:
            print(f"Fehler bei Analyse: {e}")
        
        finally:
            if raw_audio_path and os.path.exists(raw_audio_path):
                shutil.rmtree(os.path.dirname(raw_audio_path))

    # ---------------------------------------------------------
    # 4. Ausgabe der Zusammenfassung am Ende
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" ZUSAMMENFASSUNG DER GEFUNDENEN TIMESTAMPS")
    print("=" * 60)
    
    for filename, commands in summary_data.items():
        print(f"\n🎥 {filename}")
        if not commands:
            print("   -> Keine Ziel-Befehle gefunden.")
        else:
            # Sortiere die Befehle nach Zeitstempel, damit sie chronologisch angezeigt werden
            command_items = [
                (cmd, info)
                for cmd, info in commands.items()
                if not cmd.startswith("_")
            ]
            sorted_commands = sorted(command_items, key=lambda x: x[1]["timestamp_ns"])
            for cmd, info in sorted_commands:
                raw = info.get("raw_word", "")
                score = info.get("match_score", "")
                print(f"   [{info['relative_seconds']:>5.2f}s] {cmd:<7} -> {info['timestamp_ns']} ns | {raw} ({score})")
            for warning in debug_data.get(filename, {}).get("warnings", []):
                print(f"   WARNUNG: {warning}")

    # 5. Speichern als JSON-Datei für spätere ML-Pipelines
    json_path = os.path.join(args.dir, "timestamps_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    debug_json_path = os.path.join(args.dir, "timestamps_debug.json")
    with open(debug_json_path, "w") as f:
        json.dump(debug_data, f, indent=4)
        
    print("\n" + "=" * 60)
    print(f"✅ Alle Daten wurden zusätzlich gespeichert in: \n   {json_path}")
    print(f"🔎 Debug-Kandidaten und Warnungen: \n   {debug_json_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
