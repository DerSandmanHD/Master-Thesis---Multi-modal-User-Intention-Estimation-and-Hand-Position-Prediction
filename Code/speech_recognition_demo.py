import argparse
import os
import glob
import shutil
import tempfile
import wave
import json
import numpy as np

from faster_whisper import WhisperModel
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain

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
    max_val = np.max(np.abs(pcm_float))
    if max_val > 0:
        pcm_float /= max_val
    pcm_int16 = (pcm_float * 32767).astype(np.int16)

    temp_folder = tempfile.mkdtemp()
    wav_path = os.path.join(temp_folder, "audio.wav")
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(1)        
        wf.setsampwidth(2)        
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())

    return wav_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the directory containing the .vrs files.",
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
    print("Lade Whisper Model (tiny.en)...")
    model_size = "medium.en" 
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    target_commands = {"start", "second", "done" , "third"}
    
    # Hier speichern wir alle Ergebnisse
    summary_data = {}

    print(f"\nStarte Batch-Verarbeitung für {len(vrs_files)} Dateien...")
    print("-" * 50)

    # 3. Schleife über alle Videos
    for vrs_path in sorted(vrs_files):
        filename = os.path.basename(vrs_path)
        print(f"Verarbeite: {filename} ... ", end="", flush=True)
        
        summary_data[filename] = {} # Initialisiere leeres dict für dieses Video
        
        audio_path = extract_audio_gen2(vrs_path)
        if not audio_path:
            print("Fehler (Kein Audio)")
            continue

        try:
            provider = data_provider.create_vrs_data_provider(vrs_path)
            audio_stream_id = provider.get_stream_id_from_label("mic")
            audio_starting_timestamp = provider.get_first_time_ns(audio_stream_id, TimeDomain.DEVICE_TIME)
            
            segments, _ = model.transcribe(audio_path, word_timestamps=True, vad_filter=False)
            
            s_to_ns = int(1e9)
            
            for segment in segments:
                for word_info in segment.words:
                    clean_word = word_info.word.lower().strip(' .,!?')
                    
                    if clean_word in target_commands:
                        begin_ns = int(word_info.start * s_to_ns + audio_starting_timestamp)
                        rel_sec = round(word_info.start, 2)
                        
                        # Speichere den Befehl mit seinen Timestamps
                        summary_data[filename][clean_word.upper()] = {
                            "timestamp_ns": begin_ns,
                            "relative_seconds": rel_sec
                        }
            
            print("Fertig!") # Erfolgsmeldung in der gleichen Zeile

        except Exception as e:
            print(f"Fehler bei Analyse: {e}")
        
        finally:
            if audio_path and os.path.exists(audio_path):
                shutil.rmtree(os.path.dirname(audio_path))

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
            sorted_commands = sorted(commands.items(), key=lambda x: x[1]["timestamp_ns"])
            for cmd, info in sorted_commands:
                print(f"   [{info['relative_seconds']:>5.2f}s] {cmd:<7} -> {info['timestamp_ns']} ns")

    # 5. Speichern als JSON-Datei für spätere ML-Pipelines
    json_path = os.path.join(args.dir, "timestamps_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=4)
        
    print("\n" + "=" * 60)
    print(f"✅ Alle Daten wurden zusätzlich gespeichert in: \n   {json_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()