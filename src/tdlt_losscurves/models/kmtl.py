from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..data import Curve
from ..protocols import weights_for
from .common import EPS, init_common, positive_log_objective


KMTL_LAMBDAS = {
    1: [0.999],
    2: [0.99, 0.999],
    3: [0.99, 0.999, 0.9999],
}


def _memories(curve: Curve, lambdas: list[float]) -> list[np.ndarray]:
    out = []
    for lam in lambdas:
        mem = np.zeros(len(curve.step), dtype=float)
        for i in range(1, len(mem)):
            mem[i] = lam * mem[i - 1] + curve.delta_signed[i]
        out.append(np.cumsum(mem))
    return out


def predict_kmtl(curve: Curve, model: dict) -> np.ndarray:
    params = np.asarray(model["params"], dtype=float)
    lambdas = [float(x) for x in model["lambdas"]]
    L0, A, C, alpha = params[:4]
    raw_w = np.maximum(params[4:], 1.0e-12)
    weights = raw_w / np.sum(raw_w)
    terms = _memories(curve, lambdas)
    S = np.sum([w * term for w, term in zip(weights, terms)], axis=0)
    return np.maximum(L0 + A * np.maximum(curve.T, EPS) ** (-alpha) - C * S, EPS)


def fit_kmtl(
    train: list[Curve],
    fit_indices: dict[str, np.ndarray],
    tail_weight: float,
    n_restarts: int,
    seed: int,
    m: int,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    lambdas = KMTL_LAMBDAS[m]
    pre = {c.schedule: _memories(c, lambdas) for c in train}

    def objective(p: np.ndarray) -> float:
        L0, A, C, alpha = p[:4]
        raw_w = np.maximum(p[4:], 1.0e-12)
        mix = raw_w / np.sum(raw_w)
        val = 0.0
        for c in train:
            idx = fit_indices[c.schedule]
            S = np.sum([w * term[idx] for w, term in zip(mix, pre[c.schedule])], axis=0)
            pred = L0 + A * np.maximum(c.T[idx], EPS) ** (-alpha) - C * S
            val += positive_log_objective(pred, c.loss_ema[idx], weights_for(c, idx, tail_weight))
        return val

    rows: list[dict] = []
    best: dict | None = None
    for restart in range(n_restarts):
        c0 = train[0]
        idx0 = fit_indices[c0.schedule]
        L0, A, alpha = init_common(rng, c0.T[idx0], c0.loss_ema[idx0])
        init = [L0, A, float(np.exp(rng.uniform(np.log(1.0e-4), np.log(10.0)))), alpha] + list(rng.uniform(0.2, 1.0, size=m))
        bounds = [(1.0e-6, 20.0), (1.0e-8, 1.0e5), (1.0e-10, 1.0e5), (0.01, 5.0)] + [(1.0e-6, 10.0)] * m
        res = minimize(objective, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400})
        raw_w = np.maximum(res.x[4:], 1.0e-12)
        mix = raw_w / np.sum(raw_w)
        row = {
            "method": f"KMTL-m{m}",
            "restart": restart,
            "objective": float(res.fun),
            "success": bool(res.success),
            "lambdas": "|".join(str(x) for x in lambdas),
            "weights": "|".join(f"{x:.6g}" for x in mix),
            "L0": float(res.x[0]),
            "A": float(res.x[1]),
            "C": float(res.x[2]),
            "alpha": float(res.x[3]),
        }
        rows.append(row)
        if best is None or float(res.fun) < best["objective"]:
            best = {
                "method": f"KMTL-m{m}",
                "params": [float(x) for x in res.x],
                "lambdas": lambdas,
                "objective": float(res.fun),
                "fit_target": "EMA loss",
                "selection_signal": "cosine fitting objective only",
            }
    assert best is not None
    return best, rows
