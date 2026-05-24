import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import whisper
from scipy.io import wavfile
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

def run_audio_pipeline(vrs_file_path, proband_id, task_name):
    json_filename = f"proband_{proband_id}_{task_name}_metadata.json"
    temp_audio_path = "temp_extracted_audio.wav"
    
    if not os.path.exists(vrs_file_path):
        print(f"❌ VRS-Datei nicht gefunden: {vrs_file_path}")
        return

    print(f"📦 Öffne Aria-Aufnahme: {vrs_file_path}...")
    provider = data_provider.create_vrs_data_provider(vrs_file_path)
    audio_stream_id = StreamId("231-1") # Das Mikrofon-Array der Aria Gen 2

    print("🎙️ Extrahiere Audiospur aus der VRS-Datei...")
    num_audio_samples = provider.get_num_data(audio_stream_id)
    
    if num_audio_samples == 0:
        print("❌ Keine Audiodaten in dieser VRS-Datei gefunden!")
        return

    audio_data_list = []
    # Loop über alle Audiorohdaten im Stream
    for i in range(num_audio_samples):
        audio_tuple = provider.get_audio_data_by_index(audio_stream_id, i)
        # FIX: Ohne Klammern auslesen und direkt in ein numpy-Array werfen
        audio_data = np.array(audio_tuple[0].data)
        audio_data_list.append(audio_data)

    # Kombiniere die Audioschnipsel zu einer fortlaufenden Spur
    full_audio = np.concatenate(audio_data_list, axis=0)

    # Kombiniere die Audioschnipsel zu einer fortlaufenden Spur
    full_audio = np.concatenate(audio_data_list, axis=0)

    # Kombiniere die Audioschnipsel zu einer fortlaufenden Spur
    full_audio = np.concatenate(audio_data_list, axis=0)
    
    # Standard-Aria-Samplerate ist meistens 48000 Hz oder 16000 Hz
    # Wir speichern es temporär als Standard-WAV ab
    sample_rate = 48000 
    wavfile.write(temp_audio_path, sample_rate, full_audio.astype(np.int16))
    print(f"💾 Temporäre Audiodatei erstellt: {temp_audio_path}")

    print("🧠 Lade KI-Sprachmodell (OpenAI Whisper)...")
    model = whisper.load_model("base")

    print("🗣️ Analysiere Sprachbefehle für die KI-Metadaten...")
    result = model.transcribe(temp_audio_path, language="de", word_timestamps=True)

    metadata = {
        "experiment_info": {
            "proband_id": proband_id,
            "task": task_name,
            "vrs_source": os.path.basename(vrs_file_path)
        },
        "labels": []
    }

    command_mapping = {
        "start": "START",
        "intention": "INTENTION",
        "action": "ACTION_START",
        "execute": "ACTION_EXECUTE",
        "ende": "END",
        "stop": "END"
    }

    print("\n🔍 Gefundene Sprachmarker:")
    print("-" * 60)

    for segment in result["segments"]:
        text = segment["text"].lower().strip().replace(".", "").replace(",", "")
        for command, label in command_mapping.items():
            if command in text:
                timestamp_seconds = segment["start"]
                label_entry = {
                    "timestamp_vrs_seconds": round(timestamp_seconds, 3),
                    "detected_word": command,
                    "state_label": label
                }
                metadata["labels"].append(label_entry)
                print(f"🎤 Befehl '{command}' -> {label} bei Sekunde {timestamp_seconds:.3f}")

    # JSON speichern
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    # Temporäre WAV aufräumen
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    print("-" * 60)
    print(f"🎉 JSON-Metadaten erfolgreich generiert: {json_filename}")

if __name__ == "__main__":
    # Achtung: Ändere hier den Pfad auf deine tatsächliche Test-Datei im Ordner!
    run_audio_pipeline(
        vrs_file_path="../Recordings/Test_3.vrs", 
        proband_id="001", 
        task_name="robot_interaction"
    )