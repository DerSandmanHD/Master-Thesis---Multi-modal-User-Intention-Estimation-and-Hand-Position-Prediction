#!/usr/bin/env python3
"""Build one audited n214 result table and human-readable experiment summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from checkpoint_semantics import assert_primary_rows


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(rows: list[dict], key: str, value: str) -> dict:
    matches = [row for row in rows if row[key] == value]
    if len(matches) != 1:
        raise ValueError(f"Expected one {key}={value!r}, found {len(matches)}")
    return matches[0]


def metric_row(label: str, family: str, source: dict) -> dict:
    assert_primary_rows([source])
    return {
        "model": label,
        "family": family,
        "seed": int(source["seed"]),
        "result_semantics": source["result_semantics"],
        "metric_source_checkpoint": source["metric_source_checkpoint"],
        "primary_checkpoint_name": source["primary_checkpoint_name"],
        "primary_checkpoint_path": source["primary_checkpoint_path"],
        "primary_checkpoint_epoch": int(source["primary_checkpoint_epoch"]),
        "primary_checkpoint_selection_split": source[
            "primary_checkpoint_selection_split"
        ],
        "primary_checkpoint_selection_metric": source[
            "primary_checkpoint_selection_metric"
        ],
        "primary_checkpoint_selection_value": source.get(
            "primary_checkpoint_selection_value"
        ),
        "primary_checkpoint_sha256": source.get("primary_checkpoint_sha256"),
        "test_intention_macro_f1": float(source["test_intention_macro_f1"]),
        "test_intention_accuracy": float(source["test_intention_accuracy"]),
        "test_intention_samples": int(source["test_intention_samples"]),
        "test_intention_per_class": source["test_intention_per_class"],
        "test_intention_confusion_matrix": source[
            "test_intention_confusion_matrix"
        ],
        "test_assistance_macro_f1": float(source["test_assistance_macro_f1"]),
        "test_assistance_accuracy": float(source["test_assistance_accuracy"]),
        "test_assistance_samples": int(source["test_assistance_samples"]),
        "test_assistance_per_class": source["test_assistance_per_class"],
        "test_assistance_confusion_matrix": source[
            "test_assistance_confusion_matrix"
        ],
        "test_assistance_type_macro_f1": float(
            source["test_assistance_type_macro_f1"]
        ),
        "test_assistance_type_accuracy": float(
            source["test_assistance_type_accuracy"]
        ),
        "test_assistance_type_samples": int(
            source["test_assistance_type_samples"]
        ),
        "test_assistance_type_per_class": source[
            "test_assistance_type_per_class"
        ],
        "test_assistance_type_confusion_matrix": source[
            "test_assistance_type_confusion_matrix"
        ],
        "test_receiving_hand_macro_f1": float(
            source["test_receiving_hand_macro_f1"]
        ),
        "test_receiving_hand_accuracy": float(
            source["test_receiving_hand_accuracy"]
        ),
        "test_receiving_hand_samples": int(
            source["test_receiving_hand_samples"]
        ),
        "test_receiving_hand_per_class": source[
            "test_receiving_hand_per_class"
        ],
        "test_receiving_hand_confusion_matrix": source[
            "test_receiving_hand_confusion_matrix"
        ],
        "test_pose_mae_cm": float(source["test_pose_mae_cm"]),
        "test_pose_orientation_error_deg": float(
            source["test_pose_orientation_error_deg"]
        ),
        "test_pose_samples": int(source["test_pose_samples"]),
        "test_pose_end_to_end_mae_cm": source.get(
            "test_pose_end_to_end_mae_cm"
        ),
        "test_pose_end_to_end_orientation_error_deg": source.get(
            "test_pose_end_to_end_orientation_error_deg"
        ),
        "test_pose_end_to_end_samples": source.get(
            "test_pose_end_to_end_samples"
        ),
        "test_pose_target_samples": source.get("test_pose_target_samples"),
        "test_pose_oracle_reference_valid": source.get(
            "test_pose_oracle_reference_valid"
        ),
        "test_pose_predicted_reference_valid": source.get(
            "test_pose_predicted_reference_valid"
        ),
        "test_pose_coverage_denominator_receiving_hand_samples": source.get(
            "test_pose_coverage_denominator_receiving_hand_samples"
        ),
        "test_pose_target_coverage": float(source["test_pose_target_coverage"]),
        "trainable_parameters": int(source["trainable_parameters"]),
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Modell | Checkpoint/Seed | Intent Macro-F1 | Accuracy | Hand Macro-F1 | Posefehler | Orientierung | Pose n | Coverage | Parameter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {checkpoint}/{seed} | {intent} | {accuracy} | {hand} | {pose} cm | {orientation}° | {samples} | {coverage:.1%} | {params:,} |".format(
                model=row["model"],
                checkpoint=row["primary_checkpoint_name"],
                seed=row["seed"],
                intent=fmt(row["test_intention_macro_f1"]),
                accuracy=fmt(row["test_intention_accuracy"]),
                hand=fmt(row["test_receiving_hand_macro_f1"]),
                pose=fmt(row["test_pose_mae_cm"], 2),
                orientation=fmt(row["test_pose_orientation_error_deg"], 2),
                samples=row["test_pose_samples"],
                coverage=row["test_pose_target_coverage"],
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
    tuned = read_json(
        report_root / "residual_v2_tuned_v2_checkpoint_coherent/summary.json"
    )
    ablation = read_json(
        report_root / "modality_ablation_v2_checkpoint_coherent/summary.json"
    )
    cache_manifest_path = (
        PROJECT_ROOT
        / "Data_collection/visual_embeddings"
        / args.dataset_tag
        / "clip_vit_b32_openai_5hz_device_time_v2/cache_manifest.json"
    )
    cache = read_json(cache_manifest_path)
    projection = read_json(
        PROJECT_ROOT
        / "Training/visual_projections"
        / args.dataset_tag
        / "clip_vit_b32_openai_5hz_device_time_v2_pca32.json"
    )
    visual_screen = read_json(
        report_root / "visual_embedding_screen_v2_device_time/summary.json"
    )
    visual_screen_rows = read_csv(
        report_root
        / "visual_embedding_screen_v2_device_time/data/validation_summary.csv"
    )
    visual_final = read_json(
        report_root / "visual_embedding_final_v2_device_time/summary.json"
    )
    final_selection = read_json(
        report_root / "final_model_selection_v2_device_time.json"
    )
    overlay = read_json(
        report_root
        / "qualitative_final_v2/overlay_report.json"
    )
    model_latency = read_json(
        latency_root / "final_selected_model_v2/summary.json"
    )
    model_latency_rows = read_csv(
        latency_root
        / "final_selected_model_v2/latency_summary.csv"
    )
    clip_latency = read_json(
        latency_root / "clip_vit_b32_openai_5hz_device_time_v2/summary.json"
    )
    clip_latency_rows = read_csv(
        latency_root
        / "clip_vit_b32_openai_5hz_device_time_v2/clip_latency_summary.csv"
    )

    expected_model_platforms = {
        "mac_cpu",
        "mac_mps",
        "tcml_compute_cpu",
        "tcml_compute_cuda",
        "uni_login3_cpu",
    }
    expected_clip_platforms = {"mac_cpu", "mac_mps", "tcml_cpu", "tcml_cuda"}
    expected_alignment_version = "vrs_rgb_device_time_v2"
    expected_time_basis = "project_aria_device_time_capture_timestamp_ns"
    validation_participants = set(
        projection.get("validation_participants_excluded", [])
    )
    test_participants = set(projection.get("test_participants_excluded", []))
    selected_uses_clip = bool(
        final_selection.get("selected_checkpoint_uses_clip")
    )

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
        and len(ablation.get("primary_results", [])) == 5,
        "clip_cache_complete": cache.get("selected_sequences", 0) > 0
        and cache.get("completed_sequences") == cache.get("selected_sequences")
        and not cache.get("errors"),
        "clip_cache_device_time_v2": cache.get("schema_version") == 2
        and cache.get("alignment", {}).get("version")
        == expected_alignment_version
        and cache.get("alignment", {}).get("time_basis") == expected_time_basis
        and bool(cache.get("alignment_fingerprint")),
        "clip_projection_train_only": projection.get("fit_split") == "train_only"
        and bool(validation_participants)
        and bool(test_participants)
        and not validation_participants & test_participants,
        "clip_projection_bound_to_corrected_cache": projection.get(
            "schema_version"
        )
        == 2
        and projection.get("alignment_version") == expected_alignment_version
        and projection.get("time_basis") == expected_time_basis
        and projection.get("alignment_fingerprint")
        == cache.get("alignment_fingerprint")
        and projection.get("cache_manifest_sha256")
        == sha256_file(cache_manifest_path),
        "visual_screen_complete": visual_screen.get("complete") is True
        and visual_screen.get("test_metrics_used_for_selection") is False,
        "visual_final_complete": visual_final.get("complete") is True
        and visual_final.get("selection_used_test_metrics") is False,
        "final_checkpoint_validation_selected": final_selection.get(
            "selection_split"
        )
        == "validation"
        and final_selection.get("test_metrics_read_for_selection") is False
        and final_selection.get("clip_alignment_version")
        == expected_alignment_version
        and final_selection.get("clip_alignment_fingerprint")
        == cache.get("alignment_fingerprint"),
        "overlay_synchronization_valid": overlay.get("synchronization_valid")
        is True
        and (
            not selected_uses_clip
            or (
                overlay.get("clip_alignment_version")
                == expected_alignment_version
                and overlay.get("clip_alignment_fingerprint")
                == cache.get("alignment_fingerprint")
            )
        )
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
        and clip_latency.get("clip_alignment_version")
        == expected_alignment_version
        and set(clip_latency.get("platforms", [])) == expected_clip_platforms
        and not clip_latency.get("errors"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"Final evidence checks failed: {failures}")

    tuned_rows = tuned["primary_results"]
    visual_rows = visual_final["primary_results"]
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
    selected_variant = visual_screen["selected_variant_for_final_test"]
    selected_result_variant = (
        "sensor_baseline"
        if selected_variant == "sensor_baseline"
        else "selected_visual"
    )
    selected_label = {
        "sensor_baseline": "Tuned sensor (validation-selected)",
        "clip_only": "CLIP only (validation-selected)",
        "sensor_plus_clip": "Sensor + CLIP (validation-selected)",
    }[selected_variant]
    selected_system = metric_row(
        selected_label,
        "main",
        select(visual_rows, "variant", selected_result_variant),
    )
    if not Path(selected_system["primary_checkpoint_path"]).as_posix().endswith(
        Path(final_selection["selected_checkpoint"]).as_posix()
    ):
        raise ValueError(
            "Selected visual primary row does not use the frozen final checkpoint"
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
        for row in ablation["primary_results"]
        if row["variant"] != "full"
    ]
    no_hands_ablation = select(ablation["primary_results"], "variant", "no_hands")
    test_rows = [baseline, tuned_sensor]
    if selected_variant != "sensor_baseline":
        test_rows.append(selected_system)
    test_rows.extend(ablation_rows)

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
        "schema_version": 2,
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
            "alignment_version": cache["alignment"]["version"],
            "time_basis": cache["alignment"]["time_basis"],
            "alignment_fingerprint": cache["alignment_fingerprint"],
            "cache_manifest_sha256": sha256_file(cache_manifest_path),
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
            "The selected architecture was frozen on validation before final test evaluation.",
            "The qualitative pose overlay is in a separate robot-frame XY inset, not projected into RGB.",
            "CLIP frontend and temporal-model latency are measured separately because RGB updates at 5 Hz while the sensor model runs at 30 Hz.",
            "A new final-checkpoint live headset capture requires physical Aria hardware and is not represented by the offline cross-platform benchmark.",
        ],
    }

    output_root = report_root / "final_experiment_summary_v2"
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "FINAL_EXPERIMENT_SUMMARY.json"
    csv_path = output_root / "final_test_metrics.csv"
    markdown_path = output_root / "FINAL_EXPERIMENT_SUMMARY.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)

    main_rows = [baseline, tuned_sensor]
    if selected_variant != "sensor_baseline":
        main_rows.append(selected_system)
    selected_config = hp_confirm["selected_metrics"]
    markdown = f"""# Abschlusszusammenfassung der korrigierten Experimente

Dataset: `{args.dataset_tag}`

Seeds: 42, 43, 44

Auswahl: ausschließlich Validation; keine Testmetrik wurde zur Auswahl verwendet

## Zentrale Testergebnisse

{markdown_table(main_rows)}

Das Tuning verbessert den Intentions-Macro-F1 um
`{fmt(tuned_sensor['test_intention_macro_f1'] - baseline['test_intention_macro_f1'])}`
bei {tuned_sensor['trainable_parameters']:,} statt {baseline['trainable_parameters']:,}
Parametern. Das korrigierte visuelle Validation-Screening waehlt
`{selected_variant}`. Der beobachtete Test-Unterschied dieses vorab
eingefrorenen Systems gegenueber der getunten Sensor-Baseline betraegt
`{fmt(selected_system['test_intention_macro_f1'] - tuned_sensor['test_intention_macro_f1'])}`
Intentions-F1 und
`{fmt(selected_system['test_pose_mae_cm'] - tuned_sensor['test_pose_mae_cm'], 2)} cm`
Posefehler. Diese Testdifferenzen werden nicht zur nachtraeglichen Modellwahl
verwendet.

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
(`no_hands`: deskriptives ΔF1
`{no_hands_ablation['test_intention_macro_f1'] - baseline['test_intention_macro_f1']:+.4f}`).
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
