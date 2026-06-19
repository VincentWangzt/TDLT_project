from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdlt_losscurves.plotting import plot_error_comparison, plot_wsd_rmse_bar
from tdlt_losscurves.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate all curated results.")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--n-restarts", type=int, default=20)
    parser.add_argument("--fsl-restarts", type=int, default=10)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    out = ensure_dir(args.out_dir)
    run([
        sys.executable,
        "scripts/run_baselines.py",
        "--out-dir",
        str(out / "baselines"),
        "--n-restarts",
        str(args.n_restarts),
        "--fsl-restarts",
        str(args.fsl_restarts),
    ])
    run([sys.executable, "scripts/run_variants.py", "--out-dir", str(out / "variants"), "--n-restarts", str(args.n_restarts)])
    run([sys.executable, "scripts/run_tuned_mtl.py", "--out-dir", str(out / "tuned_mtl")])

    metrics = pd.concat(
        [
            pd.read_csv(out / "baselines" / "metrics.csv"),
            pd.read_csv(out / "variants" / "metrics.csv"),
            pd.read_csv(out / "tuned_mtl" / "metrics.csv"),
        ],
        ignore_index=True,
    )
    predictions = pd.concat(
        [
            pd.read_csv(out / "baselines" / "predictions.csv"),
            pd.read_csv(out / "variants" / "predictions.csv"),
            pd.read_csv(out / "tuned_mtl" / "predictions.csv"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(out / "all_metrics.csv", index=False)
    predictions.to_csv(out / "all_predictions.csv", index=False)

    fig_dir = ensure_dir(out / "figures")
    plot_wsd_rmse_bar(metrics, fig_dir / "all_wsd_rmse_bar.png")
    plot_error_comparison(predictions, "wsd", fig_dir / "all_wsd_error_comparison.png")
    main = metrics[
        (metrics["schedule"] == "wsd")
        & (metrics["target"] == "ema")
        & (metrics["region"] == "full")
    ][["method", "r2", "mae", "rmse", "finale", "worste", "mape"]].sort_values("rmse")
    print("\nHeadline WSD EMA full metrics:")
    print(main.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
