from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdlt_losscurves.experiments import run_tuned_mtl
from tdlt_losscurves.protocols import ProtocolConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freshly fit tuned MTL using the selected autoresearch setting.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "loss_curves"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "tuned_mtl"))
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ProtocolConfig()
    metrics, _ = run_tuned_mtl(args.data_root, args.out_dir, cfg, args.config_dir)
    main_rows = metrics[
        (metrics["schedule"] == "wsd")
        & (metrics["target"] == "ema")
        & (metrics["region"] == "full")
    ][["method", "rmse", "mae", "r2", "finale"]].sort_values("rmse")
    print(main_rows.to_string(index=False))
    print("\nDisclosure: tuned MTL parameters are refit from cosine data; the search setting was selected through WSD-adaptive autoresearch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
