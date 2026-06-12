#!/usr/bin/env python3
"""Run the Phase 0 deterministic smoke benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from nano_serve.benchmark.phase0 import Phase0SmokeConfig, run_phase0_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase0"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()

    summary = run_phase0_smoke(
        Phase0SmokeConfig(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            load_model=args.load_model,
        )
    )
    print(summary["artifacts"]["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
