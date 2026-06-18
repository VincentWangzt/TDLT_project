from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdlt_losscurves.experiments import run_variants
from tdlt_losscurves.protocols import ProtocolConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KMTL, FSL-MPL+, and NCPL-lite variants.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "loss_curves"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "variants"))
    parser.add_argument("--n-restarts", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ProtocolConfig(n_restarts=args.n_restarts)
    metrics, _ = run_variants(args.data_root, args.out_dir, cfg)
    main_rows = metrics[
        (metrics["schedule"] == "wsd")
        & (metrics["target"] == "ema")
        & (metrics["region"] == "full")
    ][["method", "rmse", "mae", "r2", "finale"]].sort_values("rmse")
    print(main_rows.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
