from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdlt_losscurves.experiments import run_baselines
from tdlt_losscurves.protocols import ProtocolConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MTL, MPL, and faithful FSL baselines.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "loss_curves"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "baselines"))
    parser.add_argument("--n-restarts", type=int, default=20)
    parser.add_argument("--fsl-restarts", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ProtocolConfig(n_restarts=args.n_restarts)
    metrics, _ = run_baselines(args.data_root, args.out_dir, cfg, fsl_restarts=args.fsl_restarts)
    main_rows = metrics[
        (metrics["schedule"] == "wsd")
        & (metrics["target"] == "ema")
        & (metrics["region"] == "full")
    ][["method", "rmse", "mae", "r2", "finale"]].sort_values("rmse")
    print(main_rows.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
