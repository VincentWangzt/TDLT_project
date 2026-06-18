from __future__ import annotations

import numpy as np


EPS = 1.0e-12
HUBER_DELTA = 1.0e-3


def huber(x: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    ax = np.abs(x)
    return np.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))


def positive_log_objective(pred: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
        return 1.0e30
    residual = np.log(pred) - np.log(y)
    if not np.all(np.isfinite(residual)):
        return 1.0e30
    return float(np.sum(weights * huber(residual)))


def init_common(rng: np.random.Generator, T: np.ndarray, loss: np.ndarray) -> tuple[float, float, float]:
    min_loss = float(np.min(loss))
    first_loss = float(loss[0])
    l0 = rng.uniform(max(1.0e-3, min_loss * 0.65), max(min_loss * 0.98, 1.0e-3))
    alpha = float(np.exp(rng.uniform(np.log(0.15), np.log(1.5))))
    A = max((first_loss - l0) * (T[0] ** alpha), 1.0e-4)
    return l0, A, alpha
