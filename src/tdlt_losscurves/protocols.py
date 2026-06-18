from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Curve


@dataclass(frozen=True)
class ProtocolConfig:
    ema_span: int = 201
    min_step: int = 1000
    fit_stride: int = 800
    tail_fit_stride: int = 250
    eval_stride: int = 20
    source_stride: int = 100
    eps_T: float = 1.0
    tail_weight: float = 2.0
    n_restarts: int = 20
    seed: int = 20260527
    fit_schedule: str = "cosine"
    test_schedule: str = "wsd"
    aux_schedule: str = "811"


def select_fit_idx(
    curve: Curve,
    min_step: int,
    stride: int,
    tail_stride: int | None = None,
    end_frac: float = 1.0,
) -> np.ndarray:
    max_step = curve.step[-1] * end_frac
    valid = np.flatnonzero((curve.step >= min_step) & (curve.step <= max_step))
    if len(valid) == 0:
        raise ValueError(f"No valid fit points for {curve.schedule}")
    idx = valid[:: max(1, int(stride))]
    if tail_stride is not None:
        tail_start = curve.step[valid[0]] + 0.8 * (curve.step[valid[-1]] - curve.step[valid[0]])
        tail = valid[curve.step[valid] >= tail_start][:: max(1, int(tail_stride))]
        idx = np.unique(np.concatenate([idx, tail]))
    if idx[-1] != valid[-1]:
        idx = np.unique(np.concatenate([idx, [valid[-1]]]))
    return idx.astype(np.int64)


def select_eval_idx(curve: Curve, min_step: int, stride: int, start_frac: float = 0.0) -> np.ndarray:
    start_step = max(min_step, curve.step[-1] * start_frac)
    valid = np.flatnonzero(curve.step >= start_step)
    if len(valid) == 0:
        raise ValueError(f"No valid eval points for {curve.schedule}")
    idx = valid[:: max(1, int(stride))]
    if idx[-1] != valid[-1]:
        idx = np.concatenate([idx, [valid[-1]]])
    return idx.astype(np.int64)


def strict_indices(curves: dict[str, Curve], cfg: ProtocolConfig) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fit_indices = {
        cfg.fit_schedule: select_fit_idx(
            curves[cfg.fit_schedule],
            min_step=cfg.min_step,
            stride=cfg.fit_stride,
            tail_stride=cfg.tail_fit_stride,
        )
    }
    eval_indices = {
        name: select_eval_idx(curves[name], min_step=cfg.min_step, stride=cfg.eval_stride)
        for name in [cfg.fit_schedule, cfg.test_schedule, cfg.aux_schedule]
        if name in curves
    }
    return fit_indices, eval_indices


def weights_for(curve: Curve, idx: np.ndarray, tail_weight: float) -> np.ndarray:
    weights = np.ones(len(idx), dtype=float)
    tail_start = curve.step[idx[0]] + 0.8 * (curve.step[idx[-1]] - curve.step[idx[0]])
    weights[curve.step[idx] >= tail_start] *= tail_weight
    return weights
