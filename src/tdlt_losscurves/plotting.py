from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import Curve


def plot_lr_schedules(curves: dict[str, Curve], out: str | Path) -> None:
    plt.figure(figsize=(8.0, 4.5))
    for name in ["cosine", "wsd", "811"]:
        c = curves[name]
        plt.plot(c.step, c.lr, label=name)
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.title("Course learning-rate schedules")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_loss_curves(curves: dict[str, Curve], out: str | Path) -> None:
    plt.figure(figsize=(8.0, 4.5))
    for name in ["cosine", "wsd", "811"]:
        c = curves[name]
        plt.plot(c.step, c.loss_ema, label=f"{name} EMA")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("EMA-smoothed course loss curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_prediction(predictions: pd.DataFrame, method: str, schedule: str, out: str | Path, title: str) -> None:
    sub = predictions[(predictions["method"] == method) & (predictions["schedule"] == schedule)]
    if len(sub) == 0:
        return
    plt.figure(figsize=(8.0, 4.5))
    plt.plot(sub["step"], sub["loss_ema"], label="true EMA", linewidth=1.6)
    plt.plot(sub["step"], sub["pred_loss"], "--", label="prediction", linewidth=1.4)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_error_comparison(predictions: pd.DataFrame, schedule: str, out: str | Path) -> None:
    sub = predictions[predictions["schedule"] == schedule]
    if len(sub) == 0:
        return
    plt.figure(figsize=(8.2, 4.8))
    for method, df in sub.groupby("method"):
        plt.plot(df["step"], df["error_ema"], label=method, linewidth=1.0)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("step")
    plt.ylabel("EMA prediction error")
    plt.title(f"{schedule.upper()} prediction error comparison")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_wsd_rmse_bar(metrics: pd.DataFrame, out: str | Path) -> None:
    main = metrics[
        (metrics["schedule"] == "wsd")
        & (metrics["target"] == "ema")
        & (metrics["region"] == "full")
    ].copy()
    if len(main) == 0:
        return
    main = main.sort_values("rmse")
    plt.figure(figsize=(8.2, 4.8))
    x = np.arange(len(main))
    plt.bar(x, main["rmse"])
    plt.xticks(x, main["method"], rotation=25, ha="right")
    plt.ylabel("WSD EMA full RMSE")
    plt.title("Strict cosine-fit -> WSD-test headline metric")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
