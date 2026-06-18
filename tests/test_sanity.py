from __future__ import annotations

import numpy as np
import pandas as pd

from tdlt_losscurves.data import Curve
from tdlt_losscurves.metrics import recompute_rmse_from_predictions
from tdlt_losscurves.models.fsl import fsl_sanity_checks
from tdlt_losscurves.models.kmtl import predict_kmtl
from tdlt_losscurves.models.mtl import mtl_memory, predict_mtl


def make_curve(lr: np.ndarray) -> Curve:
    eta = lr / max(float(np.max(lr)), 1.0e-12)
    delta_signed = np.concatenate([[0.0], eta[:-1] - eta[1:]])
    loss = np.linspace(4.0, 3.0, len(lr))
    return Curve(
        key="synthetic",
        schedule="cosine",
        model_size="tiny",
        dataset_size="tiny",
        step=np.arange(len(lr)),
        lr=lr,
        eta=eta,
        T=1.0 + np.cumsum(eta),
        loss_raw=loss,
        loss_ema=loss,
        delta_signed=delta_signed,
        delta_pos=np.maximum(delta_signed, 0.0),
        eta0=float(np.max(lr)),
    )


def test_constant_lr_has_zero_mtl_annealing() -> None:
    curve = make_curve(np.ones(128) * 1.0e-3)
    assert np.allclose(mtl_memory(curve, 0.999), 0.0)


def test_kmtl_m1_matches_mtl_kernel() -> None:
    curve = make_curve(np.linspace(1.0e-3, 1.0e-4, 128))
    params = [2.0, 3.0, 0.01, 0.7]
    mtl = {"method": "MTL", "lambda": 0.999, "params": params}
    kmtl = {"method": "KMTL-m1", "lambdas": [0.999], "params": params + [1.0]}
    assert np.allclose(predict_mtl(curve, mtl), predict_kmtl(curve, kmtl))


def test_fsl_zero_decay_response_and_monotonicity() -> None:
    checks = fsl_sanity_checks()
    assert checks["zero_decay_response_matches_signal"]
    assert checks["response_monotone"]
    assert checks["all_passed"]


def test_metrics_recompute_from_predictions() -> None:
    df = pd.DataFrame(
        {
            "method": ["M"] * 3,
            "schedule": ["wsd"] * 3,
            "role": ["test"] * 3,
            "loss_ema": [1.0, 2.0, 3.0],
            "pred_loss": [1.0, 2.5, 2.0],
        }
    )
    expected = float(np.sqrt(np.mean(np.array([0.0, 0.5, -1.0]) ** 2)))
    assert recompute_rmse_from_predictions(df, "M") == expected
