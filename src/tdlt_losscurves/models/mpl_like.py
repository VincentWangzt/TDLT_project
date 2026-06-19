from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..data import Curve
from ..protocols import weights_for
from .common import EPS, init_common, positive_log_objective


def compressed_sources(curve: Curve, source_stride: int, positive_delta: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delta = curve.delta_pos if positive_delta else curve.delta_signed
    ids, dsum, eta, T = [], [], [], []
    for start in range(1, len(delta), max(1, int(source_stride))):
        end = min(len(delta), start + max(1, int(source_stride)))
        amount = float(np.sum(delta[start:end]))
        if abs(amount) < 1.0e-14:
            continue
        j = end - 1
        ids.append(j)
        dsum.append(amount)
        eta.append(curve.eta[j])
        T.append(curve.T[j])
    return np.asarray(ids, dtype=np.int64), np.asarray(dsum), np.asarray(eta), np.asarray(T)


def response_sum(
    curve: Curve,
    idx: np.ndarray,
    kind: str,
    nonlin: dict,
    source_stride: int,
    source: bool = False,
) -> np.ndarray:
    _, d, eta_s, T_s = compressed_sources(curve, source_stride=source_stride, positive_delta=True)
    if len(d) == 0:
        return np.zeros(len(idx), dtype=float)
    Tt = curve.T[idx][:, None]
    raw_gap = Tt - T_s[None, :]
    mask = raw_gap >= 0.0
    gap = np.where(mask, raw_gap, 0.0)
    if kind == "mpl":
        gap = gap + eta_s[None, :]
        x = np.maximum(eta_s[None, :], 1.0e-6) ** (-nonlin["gamma"]) * gap
        resp = 1.0 - np.power(1.0 + nonlin["C"] * x, -nonlin["beta"])
    else:
        x = np.maximum(eta_s[None, :], 1.0e-6) ** (-nonlin["gamma"]) * gap
        resp = 1.0 - np.power(1.0 + nonlin["kappa"] * x, -nonlin["q"])
    resp = np.where(mask, resp, 0.0)
    if source:
        src = 1.0 + nonlin["cs"] * np.maximum(T_s, EPS) ** (-nonlin["alpha"])
    else:
        src = 1.0
    return np.sum(d[None, :] * src * resp, axis=1)


def predict_mpl_like(curve: Curve, idx: np.ndarray, model: dict, source_stride: int) -> np.ndarray:
    name = model["method"]
    params = np.asarray(model["params"], dtype=float)
    if name == "MPL":
        L0, A, B, C, alpha, beta, gamma = params
        rsum = response_sum(curve, idx, "mpl", {"C": C, "beta": beta, "gamma": gamma}, source_stride)
        return np.maximum(L0 + A * np.maximum(curve.T[idx], EPS) ** (-alpha) - B * rsum, EPS)
    if name == "FSL-MPL+ small":
        L0, A, B, alpha, kappa, gamma, q = params
        rsum = response_sum(curve, idx, "fsl", {"kappa": kappa, "gamma": gamma, "q": q}, source_stride)
        return np.maximum(L0 + A * np.maximum(curve.T[idx], EPS) ** (-alpha) - B * rsum, EPS)
    if name == "FSL-MPL+ source":
        L0, A, B, alpha, kappa, gamma, q, cs = params
        rsum = response_sum(
            curve,
            idx,
            "fsl",
            {"kappa": kappa, "gamma": gamma, "q": q, "alpha": alpha, "cs": cs},
            source_stride,
            source=True,
        )
        return np.maximum(L0 + A * np.maximum(curve.T[idx], EPS) ** (-alpha) - B * rsum, EPS)
    raise ValueError(f"unknown MPL-like model {name!r}")


def fit_mpl_like(
    model_name: str,
    train: list[Curve],
    fit_indices: dict[str, np.ndarray],
    tail_weight: float,
    n_restarts: int,
    seed: int,
    source_stride: int,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)

    def bounds_for() -> list[tuple[float, float]]:
        if model_name == "MPL":
            return [(1.0e-6, 20.0), (1.0e-8, 1.0e5), (1.0e-9, 1.0e5), (1.0e-6, 10.0), (0.01, 3.0), (0.01, 3.0), (0.01, 3.0)]
        if model_name == "FSL-MPL+ small":
            return [(1.0e-6, 20.0), (1.0e-8, 1.0e5), (1.0e-9, 1.0e5), (0.01, 3.0), (1.0e-7, 10.0), (0.05, 3.0), (0.05, 3.0)]
        return [(1.0e-6, 20.0), (1.0e-8, 1.0e5), (1.0e-9, 1.0e5), (0.01, 3.0), (1.0e-7, 10.0), (0.05, 3.0), (0.05, 3.0), (0.0, 10.0)]

    def random_init() -> list[float]:
        c0 = train[0]
        idx0 = fit_indices[c0.schedule]
        L0, A, alpha = init_common(rng, c0.T[idx0], c0.loss_ema[idx0])
        B = float(np.exp(rng.uniform(np.log(1.0e-2), np.log(5.0e2))))
        if model_name == "MPL":
            return [L0, A, B, float(np.exp(rng.uniform(np.log(1.0e-4), np.log(2.0)))), alpha, rng.uniform(0.15, 1.2), rng.uniform(0.3, 1.5)]
        if model_name == "FSL-MPL+ small":
            return [L0, A, B, alpha, float(np.exp(rng.uniform(np.log(1.0e-5), np.log(0.5)))), rng.uniform(0.5, 1.5), rng.uniform(0.2, 1.2)]
        return [L0, A, B, alpha, float(np.exp(rng.uniform(np.log(1.0e-5), np.log(0.5)))), rng.uniform(0.5, 1.5), rng.uniform(0.2, 1.2), rng.uniform(0.0, 2.0)]

    def objective(params: np.ndarray) -> float:
        val = 0.0
        model = {"method": model_name, "params": [float(x) for x in params]}
        for c in train:
            idx = fit_indices[c.schedule]
            pred = predict_mpl_like(c, idx, model, source_stride)
            val += positive_log_objective(pred, c.loss_ema[idx], weights_for(c, idx, tail_weight))
        return val + 1.0e-8 * float(np.sum(np.asarray(params, dtype=float) ** 2))

    rows: list[dict] = []
    best: dict | None = None
    for restart in range(n_restarts):
        res = minimize(objective, random_init(), method="L-BFGS-B", bounds=bounds_for(), options={"maxiter": 220, "ftol": 1.0e-10})
        row = {"method": model_name, "restart": restart, "objective": float(res.fun), "success": bool(res.success)}
        for i, value in enumerate(res.x):
            row[f"p{i}"] = float(value)
        rows.append(row)
        if best is None or float(res.fun) < best["objective"]:
            best = {
                "method": model_name,
                "params": [float(x) for x in res.x],
                "objective": float(res.fun),
                "source_stride": int(source_stride),
                "fit_target": "EMA loss",
                "selection_signal": "cosine fitting objective only",
            }
    assert best is not None
    return best, rows


def _residual_features(curve: Curve, idx: np.ndarray) -> np.ndarray:
    dpos = np.cumsum(curve.delta_pos)
    ratio = (curve.step[idx] - curve.step[idx[0]]) / max(curve.step[idx[-1]] - curve.step[idx[0]], 1)
    decay = (curve.lr[idx] < 0.999 * curve.eta0).astype(float)
    return np.column_stack(
        [
            np.log(np.maximum(curve.T[idx], EPS)),
            curve.eta[idx],
            dpos[idx],
            ratio,
            decay,
            np.maximum(curve.T[idx], EPS) ** -0.5,
        ]
    )


def fit_ncpl_lite(base_model: dict, train_curve: Curve, train_idx: np.ndarray, source_stride: int) -> dict:
    base_pred = predict_mpl_like(train_curve, train_idx, base_model, source_stride)
    residual = np.log(train_curve.loss_ema[train_idx]) - np.log(np.maximum(base_pred, EPS))
    X = _residual_features(train_curve, train_idx)
    scaler = StandardScaler().fit(X)
    best: dict | None = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha).fit(scaler.transform(X), residual)
        pred = reg.predict(scaler.transform(X))
        mse = float(np.mean((residual - pred) ** 2))
        if best is None or mse < best["mse"]:
            best = {
                "method": "FSL-MPL+ source + NCPL-lite",
                "base": base_model,
                "alpha": float(alpha),
                "coef": [float(x) for x in reg.coef_],
                "intercept": float(reg.intercept_),
                "mean": [float(x) for x in scaler.mean_],
                "scale": [float(x) for x in scaler.scale_],
                "mse": mse,
                "fit_target": "EMA log-residual on cosine",
                "selection_signal": "cosine residual fit only",
            }
    assert best is not None
    return best


def predict_ncpl_lite(model: dict, curve: Curve, idx: np.ndarray, source_stride: int) -> np.ndarray:
    base = predict_mpl_like(curve, idx, model["base"], source_stride)
    X = _residual_features(curve, idx)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coef = np.asarray(model["coef"], dtype=float)
    z = (X - mean) / np.maximum(scale, EPS)
    resid = model["intercept"] + z @ coef
    resid = np.clip(resid, -0.02, 0.02)
    return np.maximum(base * np.exp(resid), EPS)
