#!/usr/bin/env python3
"""Short checks for explicit START-at-video-beginning protocol defaults."""

from __future__ import annotations

from apply_manual_reviews import apply_video_beginning_start_defaults


def command(relative_seconds: float, audio_start_ns: int) -> dict:
    return {
        "timestamp_ns": int(audio_start_ns + relative_seconds * 1e9),
        "relative_seconds": relative_seconds,
        "timestamp_source": "automatic",
    }


def main() -> int:
    audio_start_ns = 10_000_000_000
    timestamps = {
        "Isa_1_20260730_181722.vrs": {
            "SECOND": command(5.0, audio_start_ns),
            "DONE": command(10.0, audio_start_ns),
            "THIRD": command(15.0, audio_start_ns),
        },
        "Paul_1_20260730_195810.vrs": {
            "SECOND": command(4.0, audio_start_ns),
        },
        "Isa_2_20260730_181847.vrs": {
            "START": command(1.0, audio_start_ns),
            "SECOND": command(5.0, audio_start_ns),
        },
        "Other_1_20260730_120000.vrs": {
            "SECOND": command(6.0, audio_start_ns),
        },
        "Isa_3_20260730_182001.vrs": {},
    }
    debug = {
        "Isa_1_20260730_181722.vrs": {
            "audio_start_timestamp_ns": audio_start_ns,
        },
        "Paul_1_20260730_195810.vrs": {
            "audio_start_timestamp_ns": audio_start_ns,
        },
        "Isa_2_20260730_181847.vrs": {
            "audio_start_timestamp_ns": audio_start_ns,
        },
    }

    merged, overrides, report = apply_video_beginning_start_defaults(
        timestamps,
        debug,
        {},
        {"ISA", "paul"},
    )

    for key in (
        "Isa_1_20260730_181722.vrs",
        "Paul_1_20260730_195810.vrs",
    ):
        assert merged[key]["START"]["relative_seconds"] == 0.0
        assert merged[key]["START"]["timestamp_ns"] == audio_start_ns
        assert (
            merged[key]["START"]["timestamp_source"]
            == "video_beginning_protocol_default"
        )
        assert overrides[key]["START"] == {"relative_seconds": 0.0}

    # An existing timestamp is authoritative and must not be overwritten.
    assert merged["Isa_2_20260730_181847.vrs"]["START"]["relative_seconds"] == 1.0
    assert "Isa_2_20260730_181847.vrs" not in overrides

    # Unselected participants remain untouched.
    assert "START" not in merged["Other_1_20260730_120000.vrs"]
    assert "Other_1_20260730_120000.vrs" not in overrides

    # Selected sequences without a recoverable device-time origin are rejected.
    assert "START" not in merged["Isa_3_20260730_182001.vrs"]
    assert report["participants"] == ["isa", "paul"]
    assert report["applied"] == 2
    assert report["already_present"] == 1
    assert report["rejected"] == 1

    print("Manual-review protocol-default smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
