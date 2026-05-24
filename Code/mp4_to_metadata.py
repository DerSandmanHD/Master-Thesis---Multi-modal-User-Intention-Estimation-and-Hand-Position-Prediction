import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json
import subprocess
from difflib import SequenceMatcher
from faster_whisper import WhisperModel

FUZZY_THRESHOLD = 0.82
MIN_WORD_LENGTH = 4

def fuzzy_match(word: str, command: str, threshold: float) -> bool:
    word_clean = word.lower().strip().replace(".", "").replace(",", "").replace("!", "")
    command = command.lower()

    if command in word_clean:
        return True

    if len(word_clean) < MIN_WORD_LENGTH or len(command) < MIN_WORD_LENGTH:
        return False

    ratio = SequenceMatcher(None, word_clean, command).ratio()
    return ratio >= threshold

def convert_mp4_audio_to_json(mp4_file_path, proband_id, task_name,
                               timestamp_offset_seconds=0.0):
    json_filename = f"proband_{proband_id}_{task_name}_metadata.json"
    temp_audio_path = "temp_whisper_audio.wav"

    if not os.path.exists(mp4_file_path):
        print(f"Fehler: MP4-Datei nicht gefunden: {mp4_file_path}")
        return

    print(f"Extrahiere und normalisiere Audio aus: {mp4_file_path}...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", mp4_file_path,
        "-vn", "-ar", "16000", "-ac", "1",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "pcm_s16le",
        temp_audio_path
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg Fehler:\n{result.stderr}")
        return
    print("Audio normalisiert (16kHz, Mono, loudnorm)")

    print("Lade faster-whisper Modell 'small' (CPU)...")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    print("Transkribiere...")
    segments, info = model.transcribe(
        temp_audio_path,
        language="de",
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        condition_on_previous_text=False
    )
    print(f"Sprache: {info.language} (Konfidenz: {info.language_probability:.2f})")

    # Neues Mapping basierend auf den Anforderungen des Tutors
    command_mapping = {
        "start":  "START_ASSEMBLY_TASK",
        "second": "START_GAZE_FOCUS_INTENT",
        "third":  "START_HANDOVER_INTENT",
        "done":   "DONE_MODIFIER" # Dynamische Zuordnung im Loop
    }

    metadata = {
        "experiment_info": {
            "proband_id": proband_id,
            "task": task_name,
            "mp4_source": os.path.basename(mp4_file_path),
            "audio_sample_rate_hz": 16000,
            "whisper_model": "small",
            "whisper_language": info.language,
            "timestamp_offset_applied_seconds": timestamp_offset_seconds,
            "fuzzy_threshold": FUZZY_THRESHOLD,
            "description": "Combined 3 Intents: Assembly, Gaze Focus, Handover"
        },
        "labels": []
    }

    print("\nGefundene Sprachmarker (Tutor-Logik):")
    print("-" * 70)

    seen_timestamps = [] # Liste fuer Duplikat-Schutz ueber Timestamps
    done_count = 0

    for segment in segments:
        if segment.words is None:
            continue
        for word_info in segment.words:
            raw_word = word_info.word.strip()
            
            for command, label in command_mapping.items():
                if fuzzy_match(raw_word, command, FUZZY_THRESHOLD):
                    raw_ts = word_info.start
                    
                    # Duplikat-Schutz: Verhindert, dass dasselbe Wort mehrfach zaehlt
                    if any(abs(raw_ts - ts) < 2.0 for ts in seen_timestamps):
                        continue

                    # Fallunterscheidung fuer die "done"-Befehle
                    final_label = label
                    if command == "done":
                        done_count += 1
                        if done_count == 1:
                            final_label = "END_GAZE_FOCUS_INTENT"
                        elif done_count == 2:
                            final_label = "END_HANDOVER_INTENT"
                        else:
                            final_label = f"END_UNKNOWN_INTENT_{done_count}"

                    seen_timestamps.append(raw_ts)
                    calibrated_ts = raw_ts + timestamp_offset_seconds

                    entry = {
                        "timestamp_video_seconds": round(calibrated_ts, 3),
                        "raw_timestamp_seconds": round(raw_ts, 3),
                        "whisper_word": raw_word,
                        "matched_command": command,
                        "state_label": final_label,
                        "full_segment": segment.text.strip()
                    }
                    metadata["labels"].append(entry)
                    
                    display_word = f"{command} ({done_count})" if command == "done" else command
                    print(f"🎤 '{raw_word}' -> {final_label} @ {calibrated_ts:.3f}s")

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    print("-" * 70)
    print(f"Gespeichert: {json_filename}")


if __name__ == "__main__":
    convert_mp4_audio_to_json(
        mp4_file_path="../Recordings/Test_3.mp4",
        proband_id="001",
        task_name="three_intents_sequence",
        timestamp_offset_seconds=0.0
    )