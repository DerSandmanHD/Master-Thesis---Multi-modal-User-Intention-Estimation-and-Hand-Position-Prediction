import os
import glob
import subprocess

def main():
    # Pfade zu deinen beiden Ordnern
    vrs_dir = "../Data_collection/Data_vrs"
    mp4_dir = "../Data_collection/Data_mp4"

    # Prüfen, ob die Ordner existieren
    if not os.path.exists(vrs_dir):
        print(f"Fehler: Der Ordner '{vrs_dir}' wurde nicht gefunden.")
        return
    
    # Falls der MP4-Ordner (aus welchem Grund auch immer) fehlt, erstelle ihn
    if not os.path.exists(mp4_dir):
        os.makedirs(mp4_dir)

    # Alle .vrs Dateien suchen
    vrs_files = glob.glob(os.path.join(vrs_dir, "*.vrs"))
    
    if not vrs_files:
        print(f"Keine .vrs Dateien im Ordner '{vrs_dir}' gefunden.")
        return

    print(f"\nStarte Batch-Konvertierung für {len(vrs_files)} Dateien...")
    print("-" * 60)

    converted_count = 0
    skipped_count = 0

    for vrs_path in sorted(vrs_files):
        filename = os.path.basename(vrs_path)
        base_name = os.path.splitext(filename)[0]
        
        # Der geplante Output-Pfad für die .mp4 Datei
        mp4_path = os.path.join(mp4_dir, f"{base_name}.mp4")

        # 1. Logik-Check: Existiert das Video schon?
        if os.path.exists(mp4_path):
            print(f"⏩ Überspringe {filename} (existiert bereits im mp4-Ordner)")
            skipped_count += 1
            continue

        # 2. Wenn nicht: Konvertierung starten
        print(f"🔄 Konvertiere {filename} ... ", end="", flush=True)
        
        # Der Befehl, den du sonst händisch ins Terminal getippt hast
        command = ["vrs_to_mp4", "--vrs", vrs_path, "--output_video", mp4_path]
        
        try:
            # subprocess.run führt den Terminal-Befehl unsichtbar im Hintergrund aus
            # DEVNULL unterdrückt den massiven Output des Tools, damit dein Terminal sauber bleibt
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Fertig!")
            converted_count += 1
            
        except subprocess.CalledProcessError:
            print("❌ Fehler bei der Konvertierung!")
        except FileNotFoundError:
            print("\n❌ Fehler: Der Befehl 'vrs_to_mp4' wurde nicht gefunden.")
            print("Bitte stelle sicher, dass deine 'aria_conda' Umgebung aktiviert ist.")
            return

    print("-" * 60)
    print("✅ Batch-Konvertierung komplett abgeschlossen!")
    print(f"   Neu konvertiert : {converted_count}")
    print(f"   Übersprungen    : {skipped_count}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()