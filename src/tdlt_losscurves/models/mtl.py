from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..data import Curve
from ..protocols import weights_for
from ..utils import read_json
from .common import EPS, init_common, positive_log_objective


DEFAULT_LAMBDA_GRID = [0.95, 0.99, 0.995, 0.999, 0.9995]
WIDE_LAMBDA_GRID = [0.9, 0.95, 0.97, 0.99, 0.993, 0.995, 0.997, 0.999, 0.9993, 0.9995, 0.9997, 0.9999, 0.99995]


def mtl_memory(curve: Curve, lam: float) -> np.ndarray:
    mem = np.zeros(len(curve.step), dtype=float)
    for i in range(1, len(mem)):
        mem[i] = lam * mem[i - 1] + curve.delta_signed[i]
    return np.cumsum(mem)


def predict_mtl(curve: Curve, model: dict) -> np.ndarray:
    params = np.asarray(model["params"], dtype=float)
    lam = float(model["lambda"])
    L0, A, C, alpha = params
    return np.maximum(L0 + A * np.maximum(curve.T, EPS) ** (-alpha) - C * mtl_memory(curve, lam), EPS)


def fit_mtl(
    train: list[Curve],
    fit_indices: dict[str, np.ndarray],
    tail_weight: float,
    n_restarts: int,
    seed: int,
    lambda_grid: list[float] | None = None,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    grid = lambda_grid or DEFAULT_LAMBDA_GRID
    restart_rows: list[dict] = []
    best: dict | None = None

    for lam in grid:
        memories = {c.schedule: mtl_memory(c, lam) for c in train}

        def objective(p: np.ndarray) -> float:
            val = 0.0
            for c in train:
                idx = fit_indices[c.schedule]
                pred = p[0] + p[1] * np.maximum(c.T[idx], EPS) ** (-p[3]) - p[2] * memories[c.schedule][idx]
                val += positive_log_objective(pred, c.loss_ema[idx], weights_for(c, idx, tail_weight))
            return val

        for restart in range(n_restarts):
            c0 = train[0]
            idx0 = fit_indices[c0.schedule]
            L0, A, alpha = init_common(rng, c0.T[idx0], c0.loss_ema[idx0])
            init = [L0, A, float(np.exp(rng.uniform(np.log(1.0e-4), np.log(10.0)))), alpha]
            res = minimize(
                objective,
                init,
                method="L-BFGS-B",
                bounds=[(1.0e-6, 20.0), (1.0e-8, 1.0e5), (1.0e-10, 1.0e5), (0.01, 5.0)],
                options={"maxiter": 400},
            )
            row = {
                "method": "MTL",
                "lambda": lam,
                "restart": restart,
                "objective": float(res.fun),
                "success": bool(res.success),
                "L0": float(res.x[0]),
                "A": float(res.x[1]),
                "C": float(res.x[2]),
                "alpha": float(res.x[3]),
            }
            restart_rows.append(row)
            if best is None or float(res.fun) < best["objective"]:
                best = {
                    "method": "MTL",
                    "lambda": float(lam),
                    "params": [float(x) for x in res.x],
                    "objective": float(res.fun),
                    "fit_target": "EMA loss",
                    "selection_signal": "cosine fitting objective only",
                }
    assert best is not None
    return best, restart_rows


def tuned_mtl_model(config_dir: str | Path) -> dict:
    model = read_json(Path(config_dir) / "tuned_mtl_best_model.json")
    return {
        "method": "Tuned MTL",
        "base_method": model["method"],
        "lambda": float(model["lambda"]),
        "params": [float(x) for x in model["params"]],
        "objective": float(model["objective"]),
        "fit_target": "EMA loss",
        "selection_signal": "WSD-adaptive autoresearch keep/discard loop",
        "disclosure": "Pinned model replay; WSD was used adaptively for hyperparameter/model selection.",
    }
