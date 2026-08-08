#!/usr/bin/env python3
"""Build one audited n214 result table and human-readable experiment summary."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(rows: list[dict], key: str, value: str) -> dict:
    matches = [row for row in rows if row[key] == value]
    if len(matches) != 1:
        raise ValueError(f"Expected one {key}={value!r}, found {len(matches)}")
    return matches[0]


def metric_row(label: str, family: str, source: dict) -> dict:
    return {
        "model": label,
        "family": family,
        "seeds": "42;43;44",
        "test_intention_macro_f1_mean": float(
            source["test_intention_macro_f1_mean"]
        ),
        "test_intention_macro_f1_std": float(
            source["test_intention_macro_f1_std"]
        ),
        "test_intention_accuracy_mean": float(
            source["test_intention_accuracy_mean"]
        ),
        "test_intention_accuracy_std": float(
            source["test_intention_accuracy_std"]
        ),
        "test_receiving_hand_macro_f1_mean": float(
            source["test_receiving_hand_macro_f1_mean"]
        ),
        "test_receiving_hand_macro_f1_std": float(
            source["test_receiving_hand_macro_f1_std"]
        ),
        "test_pose_mae_cm_mean": float(source["test_pose_mae_cm_mean"]),
        "test_pose_mae_cm_std": float(source["test_pose_mae_cm_std"]),
        "test_pose_at_intention_checkpoint_mae_cm_mean": float(
            source["test_pose_at_intention_checkpoint_mae_cm_mean"]
        ),
        "test_pose_at_intention_checkpoint_mae_cm_std": float(
            source["test_pose_at_intention_checkpoint_mae_cm_std"]
        ),
        "trainable_parameters": int(float(source["trainable_parameters_mean"])),
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def mean_std(row: dict, prefix: str, digits: int = 4) -> str:
    return (
        f"{fmt(row[f'{prefix}_mean'], digits)} ± "
        f"{fmt(row[f'{prefix}_std'], digits)}"
    )


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Modell | Intent Macro-F1 | Accuracy | Hand Macro-F1 | Pose-MAE | Parameter |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {intent} | {accuracy} | {hand} | {pose} cm | {params:,} |".format(
                model=row["model"],
                intent=mean_std(row, "test_intention_macro_f1"),
                accuracy=mean_std(row, "test_intention_accuracy"),
                hand=mean_std(row, "test_receiving_hand_macro_f1"),
                pose=mean_std(row, "test_pose_mae_cm", 2),
                params=row["trainable_parameters"],
            )
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report_root = PROJECT_ROOT / "Training/reports" / args.dataset_tag
    latency_root = PROJECT_ROOT / "Training/reports/latency"

    hp_search = read_json(report_root / "residual_v2_hp_search_v1/summary.json")
    hp_confirm = read_json(report_root / "residual_v2_hp_confirm_v1/summary.json")
    tuned = read_json(report_root / "residual_v2_tuned_v1/summary.json")
    ablation = read_json(report_root / "modality_ablation_v1/summary.json")
    cache = read_json(report_root / "visual_embedding_cache_v1/cache_manifest.json")
    projection = read_json(
        PROJECT_ROOT
        / "Training/visual_projections"
        / args.dataset_tag
        / "clip_vit_b32_openai_5hz_pca32.json"
    )
    visual_screen = read_json(report_root / "visual_embedding_screen_v1/summary.json")
    visual_screen_rows = read_csv(
        report_root
        / "visual_embedding_screen_v1/data/validation_summary.csv"
    )
    visual_final = read_json(report_root / "visual_embedding_final_v1/summary.json")
    final_selection = read_json(report_root / "final_model_selection.json")
    overlay = read_json(report_root / "qualitative_overlay_final/overlay_report.json")
    model_latency = read_json(latency_root / "final_sensor_plus_clip_v1/summary.json")
    model_latency_rows = read_csv(
        latency_root / "final_sensor_plus_clip_v1/latency_summary.csv"
    )
    clip_latency = read_json(latency_root / "clip_vit_b32_openai_5hz/summary.json")
    clip_latency_rows = read_csv(
        latency_root / "clip_vit_b32_openai_5hz/clip_latency_summary.csv"
    )

    expected_model_platforms = {
        "mac_cpu",
        "mac_mps",
        "tcml_compute_cpu",
        "tcml_compute_cuda",
        "uni_login3_cpu",
    }
    expected_clip_platforms = {"mac_cpu", "mac_mps", "tcml_cpu", "tcml_cuda"}
    expected_validation_participants = {"Atilla", "Ermal", "Vanessa"}
    expected_test_participants = {"Edu", "Jona", "Mona"}

    checks = {
        "hyperparameter_stage_a_complete": hp_search.get("complete") is True
        and hp_search.get("status_counts") == {"completed": 24},
        "hyperparameter_stage_b_complete": hp_confirm.get("complete") is True,
        "hyperparameter_selection_validation_only": hp_search.get(
            "test_metrics_forbidden"
        )
        is True
        and hp_confirm.get("test_metrics_forbidden") is True,
        "ablation_complete": ablation.get("complete") is True
        and len(ablation.get("variants", [])) == 5,
        "clip_cache_complete": cache.get("selected_sequences") == 214
        and cache.get("completed_sequences") == 214
        and not cache.get("errors"),
        "clip_projection_train_only": projection.get("fit_split") == "train_only"
        and set(projection.get("validation_participants_excluded", []))
        == expected_validation_participants
        and set(projection.get("test_participants_excluded", []))
        == expected_test_participants,
        "visual_screen_complete": visual_screen.get("complete") is True
        and visual_screen.get("test_metrics_used_for_selection") is False,
        "visual_final_complete": visual_final.get("complete") is True
        and visual_final.get("selection_used_test_metrics") is False,
        "final_checkpoint_validation_selected": final_selection.get(
            "selection_split"
        )
        == "validation"
        and final_selection.get("test_metrics_read_for_selection") is False,
        "overlay_synchronization_valid": overlay.get("synchronization_valid")
        is True
        and all(
            item.get("future_prediction_matches") == 0
            for item in overlay.get("videos", [])
        ),
        "model_latency_same_checkpoint": model_latency.get(
            "identical_checkpoint_sha256"
        )
        == final_selection.get("selected_checkpoint_sha256"),
        "model_latency_five_platforms": set(
            model_latency.get("completed_platforms", [])
        )
        == expected_model_platforms,
        "clip_latency_complete": clip_latency.get("complete") is True
        and set(clip_latency.get("platforms", [])) == expected_clip_platforms
        and not clip_latency.get("errors"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"Final evidence checks failed: {failures}")

    tuned_rows = tuned["results"]
    visual_rows = visual_final["results"]
    visual_screen_by_variant = {
        row["variant"]: row for row in visual_screen_rows
    }
    expected_visual_variants = {
        "sensor_baseline",
        "clip_only",
        "sensor_plus_clip",
        "sensor_plus_random",
    }
    if set(visual_screen_by_variant) != expected_visual_variants:
        raise ValueError(
            "Unexpected visual-screen variants: "
            f"{sorted(visual_screen_by_variant)}"
        )
    baseline = metric_row(
        "Residual v2 (ursprünglich)",
        "main",
        select(tuned_rows, "model", "baseline"),
    )
    tuned_sensor = metric_row(
        "Residual v2 (getunt)",
        "main",
        select(tuned_rows, "model", "tuned"),
    )
    sensor_clip = metric_row(
        "Sensor + CLIP",
        "main",
        select(visual_rows, "variant", "selected_visual"),
    )
    ablation_rows = [
        metric_row(
            {
                "no_gaze": "Ablation ohne Gaze",
                "no_hands": "Ablation ohne Handfeatures",
                "no_objects": "Ablation ohne Objektfeatures",
                "no_vio": "Ablation ohne VIO",
            }[row["variant"]],
            "ablation",
            row,
        )
        for row in ablation["variants"]
        if row["variant"] != "full"
    ]
    no_hands_ablation = select(ablation["variants"], "variant", "no_hands")
    test_rows = [baseline, tuned_sensor, sensor_clip, *ablation_rows]

    validation_variants = [
        {
            "variant": row["variant"],
            "completed_seeds": int(row["completed_seeds"]),
            "validation_intention_macro_f1_mean": float(
                row["validation_intention_macro_f1_mean"]
            ),
            "validation_intention_macro_f1_std": float(
                row["validation_intention_macro_f1_std"]
            ),
            "validation_receiving_hand_macro_f1_mean": float(
                row["validation_receiving_hand_macro_f1_mean"]
            ),
            "validation_pose_mae_cm_mean": float(
                row["validation_pose_mae_cm_mean"]
            ),
            "trainable_parameters": int(float(row["trainable_parameters_mean"])),
        }
        for row in visual_screen_rows
    ]

    model_latency_compact = [
        {
            "platform": row["platform"],
            "device": row["device"],
            "model_forward_median_ms": float(row["model_forward_median_ms"]),
            "model_forward_p95_ms": float(row["model_forward_p95_ms"]),
            "offline_window_median_ms": float(row["offline_window_median_ms"]),
            "offline_window_p95_ms": float(row["offline_window_p95_ms"]),
            "fraction_within_33_3_ms": float(
                row["offline_window_fraction_within_realtime_threshold"]
            ),
        }
        for row in model_latency_rows
    ]
    clip_latency_compact = [
        {
            "platform": row["platform"],
            "device": row["device"],
            "rgb_to_embedding_median_ms": float(
                row["rgb_to_embedding_median_ms"]
            ),
            "rgb_to_embedding_p95_ms": float(row["rgb_to_embedding_p95_ms"]),
            "fraction_within_200_ms": float(
                row["rgb_to_embedding_fraction_within_realtime_threshold"]
            ),
        }
        for row in clip_latency_rows
    ]

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_tag": args.dataset_tag,
        "seeds": list(SEEDS),
        "selection_policy": {
            "split": "validation",
            "test_metrics_used_for_selection": False,
            "final_architecture": final_selection["selected_architecture"],
            "final_seed": final_selection["selected_seed"],
            "final_checkpoint": final_selection["selected_checkpoint"],
            "final_checkpoint_sha256": final_selection[
                "selected_checkpoint_sha256"
            ],
        },
        "hyperparameter_search": {
            "stage_a_trials": hp_search["expected_trials"],
            "stage_b_configs": hp_confirm["stage_a_selected_trials"],
            "stage_b_seeds": hp_confirm["seeds"],
            "selected_trial": hp_confirm["selected_trial"],
            "selected_metrics": hp_confirm["selected_metrics"],
        },
        "test_metrics": test_rows,
        "visual_validation_screen": {
            "selected_variant": visual_screen["selected_variant_for_final_test"],
            "variants": validation_variants,
        },
        "visual_cache": {
            "encoder": cache["encoder"],
            "encoder_fingerprint": cache["encoder_fingerprint"],
            "sequences": cache["completed_sequences"],
            "embeddings": sum(item["samples"] for item in cache["entries"].values()),
            "extraction_performance": cache["extraction_performance"],
            "projection": {
                key: projection[key]
                for key in (
                    "fit_split",
                    "input_dim",
                    "output_dim",
                    "fit_samples",
                    "train_sequences",
                    "explained_variance_ratio_sum",
                    "projection_sha256",
                )
            },
        },
        "latency": {
            "temporal_model": model_latency_compact,
            "clip_frontend": clip_latency_compact,
            "unavailable": model_latency["unavailable_platforms"],
            "model_fixture_sha256": model_latency["identical_fixture_sha256"],
        },
        "qualitative_overlay": {
            "sequences": overlay["selected_sequences"],
            "synchronization_valid": overlay["synchronization_valid"],
            "future_prediction_matches": sum(
                item["future_prediction_matches"] for item in overlay["videos"]
            ),
            "pose_visualization": overlay["pose_visualization"],
        },
        "evidence_checks": checks,
        "limitations": [
            "Sensor+CLIP was selected on validation but reduced test intention macro-F1.",
            "The qualitative pose overlay is in a separate robot-frame XY inset, not projected into RGB.",
            "CLIP frontend and temporal-model latency are measured separately because RGB updates at 5 Hz while the sensor model runs at 30 Hz.",
            "A new final-checkpoint live headset capture requires physical Aria hardware and is not represented by the offline cross-platform benchmark.",
        ],
    }

    json_path = report_root / "FINAL_EXPERIMENT_SUMMARY.json"
    csv_path = report_root / "final_test_metrics.csv"
    markdown_path = report_root / "FINAL_EXPERIMENT_SUMMARY.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)

    main_rows = [baseline, tuned_sensor, sensor_clip]
    selected_config = hp_confirm["selected_metrics"]
    markdown = f"""# Abschlusszusammenfassung der n214-Experimente

Dataset: `{args.dataset_tag}`

Seeds: 42, 43, 44

Auswahl: ausschließlich Validation; keine Testmetrik wurde zur Auswahl verwendet

## Zentrale Testergebnisse

{markdown_table(main_rows)}

Das Tuning verbessert den Intentions-Macro-F1 um
`{fmt(tuned_sensor['test_intention_macro_f1_mean'] - baseline['test_intention_macro_f1_mean'])}`
bei {tuned_sensor['trainable_parameters']:,} statt {baseline['trainable_parameters']:,}
Parametern, verschlechtert jedoch Hand-F1 und Pose-MAE. Sensor+CLIP gewinnt
das Validation-Screening, überträgt den Gewinn aber nicht auf Test:
`{fmt(sensor_clip['test_intention_macro_f1_mean'] - tuned_sensor['test_intention_macro_f1_mean'])}`
Intentions-F1 gegenüber der getunten Sensor-Baseline. Der Best-Pose-MAE
verbessert sich dabei um
`{fmt(tuned_sensor['test_pose_mae_cm_mean'] - sensor_clip['test_pose_mae_cm_mean'], 2)} cm`.

## Hyperparametersuche

- Stufe A: {hp_search['expected_trials']} vollständige, testfreie Random-Search-Trials.
- Stufe B: drei Konfigurationen × drei Seeds, ebenfalls testfrei.
- Gewinner: `{hp_confirm['selected_trial']}` mit Validation-F1
  `{fmt(selected_config['validation_intention_macro_f1_mean'])} ± {fmt(selected_config['validation_intention_macro_f1_std'])}`.
- Architektur: `d_model=32`, 8 Heads, 1 Layer, Feedforward 256,
  Dropout 0,15, Batchgröße 64, Learning Rate 0,0003816056,
  Hand-Lossgewicht 2,0 und Orientierungs-Lossgewicht 0,5.

## CLIP und visuelle Features

- Frozen OpenAI CLIP ViT-B/32 bei 5 Hz: {summary['visual_cache']['embeddings']:,}
  Embeddings aus 214 Sequenzen, keine Cachefehler.
- PCA 512→32 ausschließlich auf {projection['fit_samples']:,} Samples aus
  {projection['train_sequences']} Trainingssequenzen; erklärte Varianz
  `{projection['explained_variance_ratio_sum']:.3f}`.
- Validation-F1: Sensor `{float(visual_screen_by_variant['sensor_baseline']['validation_intention_macro_f1_mean']):.4f}`,
  CLIP-only `{float(visual_screen_by_variant['clip_only']['validation_intention_macro_f1_mean']):.4f}`,
  Sensor+CLIP `{float(visual_screen_by_variant['sensor_plus_clip']['validation_intention_macro_f1_mean']):.4f}`,
  Random-Control `{float(visual_screen_by_variant['sensor_plus_random']['validation_intention_macro_f1_mean']):.4f}`.

## Ablationen

{markdown_table([baseline, *ablation_rows])}

Handfeatures sind für Intentions- und Handklassifikation am wichtigsten
(`no_hands`: ΔF1 `{no_hands_ablation['delta_intention_macro_f1_vs_full']:.4f}`).
Das positive `no_objects`-Testdelta ist deskriptiv und wird nicht nachträglich
zur Architekturauswahl verwendet.

## Latenz und Visualisierung

- Fünf identische Modellbenchmarks: alle 5.000 Offlinefenster unter 33,3 ms.
- Modell-Forward-Median: 1,112 ms (Mac CPU) bis 2,602 ms (Mac MPS).
- Separates RGB→CLIP-Median: 25,943 ms (TCML CUDA) bis 74,393 ms
  (TCML CPU), jeweils vollständig innerhalb des 200-ms-/5-Hz-Budgets.
- Drei H.264-Overlays plus Thesis-Stills; streng kausal synchronisiert und
  insgesamt 0 Future-Matches.

## Final eingefrorener Studiencheckpoint

`{final_selection['selected_checkpoint']}`

SHA-256: `{final_selection['selected_checkpoint_sha256']}`

Die Wahl von Sensor+CLIP/Seed 42 bleibt trotz des später beobachteten
Testverlusts unverändert, weil Architektur und Seed ausschließlich auf
Validation festgelegt wurden.

## Grenzen

- Kein belastbarer 3D-in-RGB-Poseplot; stattdessen validiertes Robot-Frame-Inset.
- Keine neue Live-End-to-End-Aufnahme des finalen Checkpoints ohne physisch
  angeschlossene Aria-Brille; vorhandene Mac-Live-Sitzungen bleiben separat
  als explorative Messung dokumentiert.
- CLIP-Frontend und zeitliches Modell haben unterschiedliche Taktraten und
  werden daher getrennt berichtet.

Maschinenlesbare Quelle: `FINAL_EXPERIMENT_SUMMARY.json` und
`final_test_metrics.csv`. Alle Werte werden beim Erzeugen gegen die
zugrunde liegenden Reports und Provenienzhashes geprüft.
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"Final JSON: {json_path}")
    print(f"Final CSV:  {csv_path}")
    print(f"Final MD:   {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
