#!/usr/bin/env python3
"""Fast checks for recursive run discovery and ambiguity handling."""

from __future__ import annotations

import tempfile
from pathlib import Path

from compare_final_runs import dataset_fingerprint
from run_discovery import (
    discover_run_directories,
    resolve_run_directory,
)


def make_run(path: Path) -> None:
    path.mkdir(parents=True)
    for name in ("config.json", "data_metadata.json", "metrics.json"):
        (path / name).write_text("{}\n", encoding="utf-8")


def main() -> int:
    legacy_metadata = {
        "split": {
            "dataset_filter": {
                "sequence_fingerprint": "legacy",
            }
        }
    }
    assert dataset_fingerprint(legacy_metadata) == (
        "legacy",
        "legacy_sequence_ids",
    )
    current_metadata = {
        **legacy_metadata,
        "provenance": {
            "dataset_content_fingerprint": "content",
        },
    }
    assert dataset_fingerprint(current_metadata) == (
        "content",
        "content_sha256",
    )

    with tempfile.TemporaryDirectory(
        prefix="aria_run_discovery_"
    ) as directory:
        root = Path(directory)
        direct = root / "final_clean_v1_gru_seed42"
        nested = (
            root
            / "run_cluster"
            / "final_clean_v1_residual_v2_seed44"
        )
        ignored = root / "run_cluster" / "incomplete"
        make_run(direct)
        make_run(nested)
        ignored.mkdir()
        (ignored / "config.json").write_text("{}\n", encoding="utf-8")

        discovered = discover_run_directories(
            root,
            name_prefix="final_clean_v1_",
        )
        assert discovered == sorted(
            [direct.resolve(), nested.resolve()]
        )
        assert resolve_run_directory(
            Path(nested.name),
            runs_root=root,
        ) == nested.resolve()
        assert resolve_run_directory(
            direct,
            runs_root=root,
        ) == direct.resolve()

        duplicate = root / "another" / nested.name
        make_run(duplicate)
        try:
            resolve_run_directory(
                Path(nested.name),
                runs_root=root,
            )
        except ValueError as exc:
            assert "ambiguous" in str(exc)
        else:
            raise AssertionError("Ambiguous run basename was not rejected")

    print("run discovery smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
