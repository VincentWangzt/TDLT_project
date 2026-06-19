from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .data import Curve


METHOD_LABELS = {
    "Tuned MTL": "Tuned-MTL",
    "KMTL-m2": "Two-scale MTL",
    "KMTL-m3": "Three-scale MTL",
    "FSL-MPL+ small": "Compact FSL-MPL",
    "FSL-MPL+ source": "Source-weighted FSL-MPL",
    "FSL-MPL+ source + NCPL-lite": "Residual-corrected FSL-MPL",
}

METHOD_GROUPS = {
    "MTL": "Baseline",
    "MPL": "Baseline",
    "FSL": "Baseline",
    "Tuned MTL": "Main contribution",
}

GROUP_COLORS = {
    "Baseline": "#7F0000",
    "Main contribution": "#E69F00",
    "Exploratory variant": "#7A8793",
}


def display_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_group(method: str) -> str:
    return METHOD_GROUPS.get(method, "Exploratory variant")


def plot_lr_schedules(curves: dict[str, Curve], out: str | Path) -> None:
    plt.figure(figsize=(8.0, 4.5))
    labels = {"cosine": "Cosine", "wsd": "WSD", "811": "8-1-1"}
    colors = {"cosine": "#7F0000", "wsd": "#2166AC", "811": "#4D9221"}
    for name in ["cosine", "wsd", "811"]:
        c = curves[name]
        plt.plot(c.step, c.lr, label=labels[name], color=colors[name], linewidth=1.8)
    plt.xlabel("Training step")
    plt.ylabel("Learning rate")
    plt.title("Learning-rate schedules in the course data")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_loss_curves(curves: dict[str, Curve], out: str | Path) -> None:
    plt.figure(figsize=(8.0, 4.5))
    labels = {"cosine": "Cosine", "wsd": "WSD", "811": "8-1-1"}
    colors = {"cosine": "#7F0000", "wsd": "#2166AC", "811": "#4D9221"}
    styles = {"cosine": "-", "wsd": "--", "811": "-."}
    for name in ["cosine", "wsd", "811"]:
        c = curves[name]
        visible = c.step >= 1000
        plt.plot(
            c.step[visible],
            c.loss_ema[visible],
            label=labels[name],
            color=colors[name],
            linestyle=styles[name],
            linewidth=1.8,
        )
    plt.xlabel("Training step")
    plt.ylabel("EMA-smoothed loss")
    plt.ylim(2.5, 4.1)
    plt.title("EMA-smoothed loss after the warmup region")
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
    plt.plot(sub["step"], sub["loss_ema"], label="Observed EMA", linewidth=1.8, color="#222222")
    plt.plot(sub["step"], sub["pred_loss"], "--", label="Prediction", linewidth=1.8, color="#B2182B")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
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
        group = method_group(method)
        is_main = group == "Main contribution"
        plt.plot(
            df["step"],
            df["error_ema"],
            label=display_method(method),
            color=GROUP_COLORS[group],
            linewidth=2.0 if is_main else 1.0,
            alpha=1.0 if group != "Exploratory variant" else 0.55,
        )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Training step")
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
    order = [
        "MTL",
        "MPL",
        "FSL",
        "Tuned MTL",
        "FSL-MPL+ source + NCPL-lite",
        "FSL-MPL+ source",
        "KMTL-m3",
        "FSL-MPL+ small",
        "KMTL-m2",
    ]
    rank = {name: i for i, name in enumerate(order)}
    main["display_order"] = main["method"].map(rank).fillna(len(order))
    main = main.sort_values("display_order")
    plt.figure(figsize=(8.2, 4.8))
    x = np.arange(len(main))
    colors = [GROUP_COLORS[method_group(m)] for m in main["method"]]
    plt.bar(x, main["rmse"], color=colors)
    plt.xticks(x, [display_method(m) for m in main["method"]], rotation=23, ha="right")
    plt.ylabel("WSD EMA full RMSE")
    plt.title("Prediction error under the cosine-to-WSD protocol")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(
        handles=[Patch(facecolor=color, label=group) for group, color in GROUP_COLORS.items()],
        frameon=False,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
