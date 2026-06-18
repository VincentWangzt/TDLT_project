from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOSS_COL = "Metrics/loss"
LR_COL = "lr"
STEP_COL = "step"
EPS = 1.0e-12


@dataclass(frozen=True)
class Curve:
    key: str
    schedule: str
    model_size: str
    dataset_size: str
    step: np.ndarray
    lr: np.ndarray
    eta: np.ndarray
    T: np.ndarray
    loss_raw: np.ndarray
    loss_ema: np.ndarray
    delta_signed: np.ndarray
    delta_pos: np.ndarray
    eta0: float


def parse_key(key: str) -> tuple[str, str, str]:
    schedule = re.search(r"scheduler:([^_]+)", key)
    model = re.search(r"M:([^_]+)", key)
    data = re.search(r"D:([^_]+)", key)
    return (
        schedule.group(1).lower() if schedule else infer_schedule(key),
        model.group(1) if model else "unknown",
        data.group(1) if data else "unknown",
    )


def infer_schedule(key: str) -> str:
    low = key.lower()
    if "cosine" in low:
        return "cosine"
    if "wsd" in low:
        return "wsd"
    if "811" in low or "8-1-1" in low:
        return "811"
    return "unknown"


def ema(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy(dtype=float)


def scaled_lr(lr: np.ndarray, lr_scale: str) -> np.ndarray:
    lr = np.asarray(lr, dtype=float)
    if lr.ndim != 1:
        raise ValueError("lr must be one-dimensional")
    if not np.all(np.isfinite(lr)):
        raise ValueError("lr contains NaN or inf")
    if np.any(lr < 0):
        raise ValueError("lr contains negative values")
    if lr_scale == "raw":
        return lr
    if lr_scale == "max":
        peak = float(np.max(lr))
        if peak <= 0:
            raise ValueError("cannot normalize an all-zero learning-rate curve")
        return lr / peak
    raise ValueError(f"unknown lr_scale={lr_scale!r}")


def load_curves(
    data_root: str | Path,
    ema_span: int = 201,
    eps_T: float = 1.0,
    lr_scale: str = "max",
) -> dict[str, Curve]:
    """Load the course pandas pickle and normalize it into schedule-keyed curves."""

    pkl_path = Path(data_root) / "gpt_loss+lrs.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    obj = pd.read_pickle(pkl_path)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict in {pkl_path}, got {type(obj)}")

    curves: dict[str, Curve] = {}
    for key, df in obj.items():
        for col in (STEP_COL, LOSS_COL, LR_COL):
            if col not in df.columns:
                raise KeyError(f"{key} is missing required column {col}")

        clean = df[[STEP_COL, LOSS_COL, LR_COL]].copy()
        clean.columns = ["step", "loss", "lr"]
        clean = clean.apply(pd.to_numeric, errors="coerce").dropna().sort_values("step").reset_index(drop=True)

        step = clean["step"].to_numpy(dtype=np.int64)
        loss = clean["loss"].to_numpy(dtype=float)
        lr = clean["lr"].to_numpy(dtype=float)
        if np.any(loss <= 0):
            raise ValueError(f"{key} has non-positive loss values")

        schedule, model_size, dataset_size = parse_key(str(key))
        eta = scaled_lr(lr, lr_scale=lr_scale)
        eta_prev = np.concatenate([[eta[0]], eta[:-1]])
        delta_signed = eta_prev - eta
        delta_pos = np.maximum(delta_signed, 0.0)

        curves[schedule] = Curve(
            key=str(key),
            schedule=schedule,
            model_size=model_size,
            dataset_size=dataset_size,
            step=step,
            lr=lr,
            eta=eta,
            T=float(eps_T) + np.cumsum(eta),
            loss_raw=loss,
            loss_ema=ema(loss, span=ema_span),
            delta_signed=delta_signed,
            delta_pos=delta_pos,
            eta0=float(np.max(lr)),
        )
    return curves


def select_curve(curves: dict[str, Curve], schedule: str) -> Curve:
    if schedule not in curves:
        raise KeyError(f"missing schedule {schedule!r}; available={sorted(curves)}")
    return curves[schedule]


def build_manifest(curves: dict[str, Curve], fit_schedule: str = "cosine", test_schedule: str = "wsd") -> pd.DataFrame:
    rows = []
    for c in sorted(curves.values(), key=lambda x: x.schedule):
        last_n = min(1000, len(c.loss_raw))
        rows.append(
            {
                "curve_key": c.key,
                "schedule_type": c.schedule,
                "model_size": c.model_size,
                "dataset_size": c.dataset_size,
                "number_of_points": int(len(c.step)),
                "min_step": int(np.min(c.step)),
                "max_step": int(np.max(c.step)),
                "loss_column": LOSS_COL,
                "lr_column": LR_COL,
                "peak_lr": float(np.max(c.lr)),
                "final_lr": float(c.lr[-1]),
                "raw_final_loss": float(c.loss_raw[-1]),
                "ema_final_loss": float(c.loss_ema[-1]),
                "last_1000_raw_loss_std": float(np.std(c.loss_raw[-last_n:])),
                "used_in_main_fit": c.schedule == fit_schedule,
                "used_in_main_test": c.schedule == test_schedule,
                "notes": "Readme: pandas pkl; Metrics/loss is loss; lr is learning rate; 811 decays at 80% and 90%.",
            }
        )
    return pd.DataFrame(rows)
