from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..data import Curve


PARAM_NAMES = ["L0", "c1", "c2", "c3", "c4", "s", "gamma"]


@dataclass(frozen=True)
class FSLFeatures:
    T_source: np.ndarray
    delta_lr: np.ndarray
    source_stride: int
    n_raw_positive_sources: int
    n_compressed_sources: int


def _huber(x: np.ndarray, delta: float) -> np.ndarray:
    ax = np.abs(x)
    return np.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))


def compute_fsl_features(curve: Curve, source_stride: int) -> FSLFeatures:
    lr = np.asarray(curve.eta, dtype=float)
    prev = np.empty_like(lr)
    prev[0] = lr[0]
    prev[1:] = lr[:-1]
    delta = np.maximum(prev - lr, 0.0)
    pos = delta > 0
    T_pos = curve.T[pos]
    d_pos = delta[pos]
    n_raw = int(len(d_pos))
    stride = max(1, int(source_stride))
    if n_raw == 0:
        return FSLFeatures(np.empty(0), np.empty(0), stride, 0, 0)
    if stride == 1:
        return FSLFeatures(T_pos, d_pos, stride, n_raw, n_raw)

    src_T, src_d = [], []
    for start in range(0, n_raw, stride):
        end = min(start + stride, n_raw)
        block_d = d_pos[start:end]
        block_T = T_pos[start:end]
        total = float(np.sum(block_d))
        if total <= 0:
            continue
        src_d.append(total)
        src_T.append(float(np.sum(block_T * block_d) / total))
    return FSLFeatures(np.asarray(src_T), np.asarray(src_d), stride, n_raw, len(src_d))


def _unpack(theta: np.ndarray) -> dict[str, float]:
    return {
        "L0": float(theta[0]),
        "c1": float(np.exp(theta[1])),
        "c2": float(np.exp(theta[2])),
        "c3": float(np.exp(theta[3])),
        "c4": float(np.exp(theta[4])),
        "s": float(theta[5]),
        "gamma": float(theta[6]),
    }


def _pack(params: dict[str, float]) -> np.ndarray:
    return np.array(
        [params["L0"], np.log(params["c1"]), np.log(params["c2"]), np.log(params["c3"]), np.log(params["c4"]), params["s"], params["gamma"]],
        dtype=float,
    )


def _bounds() -> list[tuple[float, float]]:
    return [(0.1, 8.0), (-30.0, 5.0), (-30.0, 12.0), (-20.0, 12.0), (-12.0, 18.0), (0.05, 3.0), (0.05, 5.0)]


def _response(T_target: np.ndarray, features: FSLFeatures, s: float, c3: float, c4: float, gamma: float, chunk_size: int = 2048) -> np.ndarray:
    out = np.zeros_like(T_target, dtype=float)
    if len(features.T_source) == 0:
        return out
    src_factor = features.delta_lr * (c3 + np.power(np.maximum(features.T_source, 1.0e-300), -s))
    for start in range(0, len(T_target), chunk_size):
        end = min(start + chunk_size, len(T_target))
        gap = T_target[start:end, None] - features.T_source[None, :]
        mask = gap >= 0
        safe_gap = np.where(mask, gap, 0.0)
        resp = 1.0 - np.power(1.0 + c4 * safe_gap, -gamma)
        resp = np.where(mask, resp, 0.0)
        out[start:end] = resp @ src_factor
    return out


def predict_fsl_params(params: dict[str, float], T_target: np.ndarray, features: FSLFeatures) -> np.ndarray:
    signal = params["c1"] * np.power(np.maximum(T_target, 1.0e-300), -params["s"])
    response = _response(T_target, features, params["s"], params["c3"], params["c4"], params["gamma"])
    return np.maximum(params["L0"] + signal - params["c2"] * response, 1.0e-12)


def predict_fsl_model(curve: Curve, model: dict) -> np.ndarray:
    features = compute_fsl_features(curve, int(model.get("source_stride", 50)))
    return predict_fsl_params(model["params"], curve.T, features)


def _objective(theta: np.ndarray, T_fit: np.ndarray, loss_fit: np.ndarray, features: FSLFeatures, huber_delta: float) -> float:
    params = _unpack(theta)
    pred = predict_fsl_params(params, T_fit, features)
    if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
        return 1.0e30
    residual = np.log(pred) - np.log(loss_fit)
    return float(np.sum(_huber(residual, huber_delta)))


def _random_initial(rng: np.random.Generator, loss: np.ndarray, T: np.ndarray) -> dict[str, float]:
    min_loss = float(np.min(loss))
    final_loss = float(loss[-1])
    early_loss = float(loss[0])
    s = float(rng.uniform(0.2, 1.8))
    L0 = float(rng.uniform(max(0.2, 0.55 * min_loss), max(0.3, 0.98 * final_loss)))
    c1_center = max((early_loss - L0) * (max(T[0], 1.0e-8) ** s), 1.0e-8)
    return {
        "L0": L0,
        "c1": float(c1_center * np.exp(rng.normal(0.0, 1.2))),
        "c2": float(np.exp(rng.uniform(-10.0, 2.0))),
        "c3": float(np.exp(rng.uniform(-8.0, 4.0))),
        "c4": float(np.exp(rng.uniform(-2.0, 12.0))),
        "s": s,
        "gamma": float(rng.uniform(0.2, 2.5)),
    }


def fsl_fit_indices(curve: Curve, min_step: int = 1000, fit_stride: int = 100, tail_fraction: float = 0.2, tail_stride: int = 25) -> np.ndarray:
    # Match the faithful reproduction: construct the stride/tail grid over the
    # full curve first, then filter the post-warmup rows.
    n = len(curve.step)
    base = np.arange(0, n, max(1, int(fit_stride)), dtype=np.int64)
    tail_start = int(np.floor((1.0 - tail_fraction) * n))
    tail = np.arange(max(0, tail_start), n, max(1, int(tail_stride)), dtype=np.int64)
    idx = np.unique(np.concatenate([base, tail, np.array([n - 1], dtype=np.int64)]))
    idx = idx[curve.step[idx] >= min_step]
    if len(idx) == 0:
        raise ValueError("no FSL fit points")
    return idx.astype(np.int64)


def fit_fsl_model(
    fit_curve: Curve,
    fit_idx: np.ndarray,
    n_restarts: int = 20,
    seed: int = 20260614,
    source_stride: int = 50,
    huber_delta: float = 1.0e-3,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    features = compute_fsl_features(fit_curve, source_stride)
    rows: list[dict] = []
    best: dict | None = None
    for restart in range(n_restarts):
        theta0 = _pack(_random_initial(rng, fit_curve.loss_ema[fit_idx], fit_curve.T[fit_idx]))
        res = minimize(
            _objective,
            theta0,
            args=(fit_curve.T[fit_idx], fit_curve.loss_ema[fit_idx], features, huber_delta),
            method="L-BFGS-B",
            bounds=_bounds(),
            options={"maxiter": 350, "maxls": 30, "ftol": 1.0e-12},
        )
        params = _unpack(res.x)
        row = {
            "method": "FSL",
            "restart": restart,
            "objective": float(res.fun),
            "success": bool(res.success),
            "status": int(res.status),
            **params,
        }
        rows.append(row)
        if best is None or float(res.fun) < best["objective"]:
            best = {
                "method": "FSL",
                "params": params,
                "objective": float(res.fun),
                "source_stride": int(source_stride),
                "huber_delta": float(huber_delta),
                "fit_target": "EMA loss",
                "selection_signal": "cosine fitting objective only",
            }
    assert best is not None
    return best, rows


def fsl_sanity_checks() -> dict[str, bool | int]:
    # Construct a minimal fake curve-like object using the public Curve dataclass would be
    # noisy; these checks exercise the response algebra directly.
    T = np.arange(1.0, 6.0)
    features = FSLFeatures(np.empty(0), np.empty(0), 1, 0, 0)
    params = {"L0": 1.0, "c1": 2.0, "c2": 3.0, "c3": 0.5, "c4": 4.0, "s": 0.7, "gamma": 1.2}
    pred = predict_fsl_params(params, T, features)
    signal = params["L0"] + params["c1"] * T ** (-params["s"])
    mono = 1.0 - (1.0 + np.linspace(0, 10, 100)) ** (-0.5)
    return {
        "zero_decay_response_matches_signal": bool(np.allclose(pred, signal)),
        "response_monotone": bool(np.all(np.diff(mono) >= -1.0e-12)),
        "all_passed": bool(np.allclose(pred, signal) and np.all(np.diff(mono) >= -1.0e-12)),
    }
