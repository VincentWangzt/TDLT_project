from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Curve


EPS = 1.0e-12


def metric_values(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = pred - y
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "finale": float(abs(err[-1])),
        "relative_finale": float(abs(err[-1]) / max(abs(y[-1]), EPS)),
        "worste": float(np.max(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(y), EPS))),
        "auc_error": float(np.trapz(np.abs(err)) / max(len(err) - 1, 1)),
        "tailmean_error": float(abs(np.mean(pred) - np.mean(y))),
        "n_points": int(len(y)),
    }


def region_masks(curve: Curve, idx: np.ndarray) -> dict[str, np.ndarray]:
    steps = curve.step[idx]
    full = np.ones(len(idx), dtype=bool)
    tail20 = steps >= steps[0] + 0.8 * (steps[-1] - steps[0])
    final1000 = steps >= steps[-1] - 1000
    decay = curve.lr[idx] < 0.999 * np.max(curve.lr)
    if curve.schedule == "cosine":
        decay = tail20.copy()
    stable = ~decay
    return {"full": full, "tail20": tail20, "decay": decay, "final1000": final1000, "stable": stable}


def evaluate_regions(
    curve: Curve,
    idx: np.ndarray,
    pred: np.ndarray,
    method: str,
    protocol: str,
    role: str,
) -> list[dict]:
    rows: list[dict] = []
    for target, y in [("ema", curve.loss_ema[idx]), ("raw", curve.loss_raw[idx])]:
        for region, mask in region_masks(curve, idx).items():
            if int(np.sum(mask)) < 3:
                continue
            row = {
                "protocol": protocol,
                "method": method,
                "schedule": curve.schedule,
                "role": role,
                "target": target,
                "region": region,
            }
            row.update(metric_values(y[mask], pred[mask]))
            rows.append(row)
    return rows


def prediction_rows(
    curve: Curve,
    idx: np.ndarray,
    pred: np.ndarray,
    method: str,
    protocol: str,
    role: str,
) -> list[dict]:
    rows = []
    for j, yhat in zip(idx, pred):
        rows.append(
            {
                "protocol": protocol,
                "method": method,
                "schedule": curve.schedule,
                "role": role,
                "step": int(curve.step[j]),
                "lr": float(curve.lr[j]),
                "loss_raw": float(curve.loss_raw[j]),
                "loss_ema": float(curve.loss_ema[j]),
                "pred_loss": float(yhat),
                "error_ema": float(yhat - curve.loss_ema[j]),
                "error_raw": float(yhat - curve.loss_raw[j]),
            }
        )
    return rows


def recompute_rmse_from_predictions(predictions: pd.DataFrame, method: str, schedule: str = "wsd") -> float:
    mask = (
        (predictions["method"] == method)
        & (predictions["schedule"] == schedule)
        & (predictions["role"].isin(["test", "extra_unseen", "fit"]))
    )
    sub = predictions.loc[mask]
    if len(sub) == 0:
        raise ValueError(f"no prediction rows for {method=} {schedule=}")
    err = sub["pred_loss"].to_numpy(dtype=float) - sub["loss_ema"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(err**2)))
