from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import Curve, build_manifest, load_curves
from .metrics import evaluate_regions, prediction_rows
from .models.fsl import fsl_fit_indices, fsl_sanity_checks, fit_fsl_model, predict_fsl_model
from .models.kmtl import fit_kmtl, predict_kmtl
from .models.mpl_like import fit_mpl_like, fit_ncpl_lite, predict_mpl_like, predict_ncpl_lite
from .models.mtl import fit_mtl, predict_mtl, tuned_mtl_model
from .plotting import plot_error_comparison, plot_loss_curves, plot_lr_schedules, plot_prediction, plot_wsd_rmse_bar
from .protocols import ProtocolConfig, strict_indices
from .utils import ensure_dir, write_json


def predict_model(curve: Curve, idx: np.ndarray, model: dict, cfg: ProtocolConfig) -> np.ndarray:
    name = model["method"]
    if name in {"MTL", "Tuned MTL"}:
        return predict_mtl(curve, model)[idx]
    if name.startswith("KMTL"):
        return predict_kmtl(curve, model)[idx]
    if name == "FSL":
        return predict_fsl_model(curve, model)[idx]
    if name == "FSL-MPL+ source + NCPL-lite":
        return predict_ncpl_lite(model, curve, idx, cfg.source_stride)
    return predict_mpl_like(curve, idx, model, cfg.source_stride)


def evaluate_models(
    curves: dict[str, Curve],
    eval_indices: dict[str, np.ndarray],
    models: list[dict],
    cfg: ProtocolConfig,
    protocol: str = "strict",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    pred_rows: list[dict] = []
    for model in models:
        for schedule, idx in eval_indices.items():
            c = curves[schedule]
            role = "fit" if schedule == cfg.fit_schedule else ("test" if schedule == cfg.test_schedule else "extra_unseen")
            pred = predict_model(c, idx, model, cfg)
            metric_rows.extend(evaluate_regions(c, idx, pred, model["method"], protocol, role))
            pred_rows.extend(prediction_rows(c, idx, pred, model["method"], protocol, role))
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def write_result_bundle(
    out_dir: str | Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    models: list[dict],
    restarts: pd.DataFrame | None = None,
    manifest: pd.DataFrame | None = None,
    curves_for_plots: dict[str, Curve] | None = None,
) -> None:
    out = ensure_dir(out_dir)
    fig_dir = ensure_dir(out / "figures")
    metrics.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    write_json(out / "fitted_params.json", {"models": models})
    if restarts is not None:
        restarts.to_csv(out / "restart_summary.csv", index=False)
    if manifest is not None:
        manifest.to_csv(out / "data_manifest.csv", index=False)
    if curves_for_plots is not None:
        plot_lr_schedules(curves_for_plots, fig_dir / "lr_schedules.png")
        plot_loss_curves(curves_for_plots, fig_dir / "loss_curves_ema.png")
    plot_wsd_rmse_bar(metrics, fig_dir / "wsd_rmse_bar.png")
    plot_error_comparison(predictions, "wsd", fig_dir / "wsd_error_comparison.png")
    for method in sorted(predictions["method"].unique()):
        safe = method.lower().replace("+", "plus").replace(" ", "_").replace("-", "")
        plot_prediction(predictions, method, "wsd", fig_dir / f"{safe}_wsd_prediction.png", f"{method}: WSD prediction")
        plot_prediction(predictions, method, "cosine", fig_dir / f"{safe}_cosine_fit.png", f"{method}: cosine fit")


def run_baselines(data_root: str | Path, out_dir: str | Path, cfg: ProtocolConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = load_curves(data_root, ema_span=cfg.ema_span, eps_T=cfg.eps_T, lr_scale="max")
    fit_idx, eval_idx = strict_indices(curves, cfg)
    train = [curves[cfg.fit_schedule]]

    mtl, mtl_restarts = fit_mtl(train, fit_idx, cfg.tail_weight, cfg.n_restarts, cfg.seed)
    mpl, mpl_restarts = fit_mpl_like("MPL", train, fit_idx, cfg.tail_weight, cfg.n_restarts, cfg.seed + 101, cfg.source_stride)

    fsl_curves = load_curves(data_root, ema_span=cfg.ema_span, eps_T=1.0e-8, lr_scale="raw")
    fsl_fit = fsl_fit_indices(fsl_curves[cfg.fit_schedule])
    fsl, fsl_restarts = fit_fsl_model(fsl_curves[cfg.fit_schedule], fsl_fit, n_restarts=cfg.n_restarts, seed=20260614, source_stride=50)
    fsl_eval_idx = {
        name: eval_idx[name]
        for name in eval_idx
    }

    metrics_a, preds_a = evaluate_models(curves, eval_idx, [mtl, mpl], cfg)
    metrics_b, preds_b = evaluate_models(fsl_curves, fsl_eval_idx, [fsl], cfg)
    metrics = pd.concat([metrics_a, metrics_b], ignore_index=True)
    predictions = pd.concat([preds_a, preds_b], ignore_index=True)
    restarts = pd.concat([pd.DataFrame(mtl_restarts), pd.DataFrame(mpl_restarts), pd.DataFrame(fsl_restarts)], ignore_index=True)
    manifest = build_manifest(curves, cfg.fit_schedule, cfg.test_schedule)
    write_result_bundle(out_dir, metrics, predictions, [mtl, mpl, fsl], restarts, manifest, curves)
    write_json(Path(out_dir) / "sanity_checks.json", {"fsl": fsl_sanity_checks()})
    return metrics, predictions


def run_variants(data_root: str | Path, out_dir: str | Path, cfg: ProtocolConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = load_curves(data_root, ema_span=cfg.ema_span, eps_T=cfg.eps_T, lr_scale="max")
    fit_idx, eval_idx = strict_indices(curves, cfg)
    train = [curves[cfg.fit_schedule]]

    models: list[dict] = []
    restart_frames: list[pd.DataFrame] = []
    for offset, (name, fit_fn) in enumerate(
        [
            ("KMTL-m2", lambda: fit_kmtl(train, fit_idx, cfg.tail_weight, cfg.n_restarts, cfg.seed + 202, 2)),
            ("KMTL-m3", lambda: fit_kmtl(train, fit_idx, cfg.tail_weight, cfg.n_restarts, cfg.seed + 303, 3)),
            ("FSL-MPL+ small", lambda: fit_mpl_like("FSL-MPL+ small", train, fit_idx, cfg.tail_weight, cfg.n_restarts, cfg.seed + 404, cfg.source_stride)),
            ("FSL-MPL+ source", lambda: fit_mpl_like("FSL-MPL+ source", train, fit_idx, cfg.tail_weight, cfg.n_restarts, cfg.seed + 505, cfg.source_stride)),
        ]
    ):
        model, rows = fit_fn()
        models.append(model)
        frame = pd.DataFrame(rows)
        frame["fit_group"] = name
        frame["fit_offset"] = offset
        restart_frames.append(frame)

    source_model = next(m for m in models if m["method"] == "FSL-MPL+ source")
    ncpl = fit_ncpl_lite(source_model, curves[cfg.fit_schedule], fit_idx[cfg.fit_schedule], cfg.source_stride)
    models.append(ncpl)

    metrics, predictions = evaluate_models(curves, eval_idx, models, cfg)
    restarts = pd.concat(restart_frames, ignore_index=True)
    manifest = build_manifest(curves, cfg.fit_schedule, cfg.test_schedule)
    write_result_bundle(out_dir, metrics, predictions, models, restarts, manifest, curves)
    return metrics, predictions


def run_tuned_mtl(data_root: str | Path, out_dir: str | Path, cfg: ProtocolConfig, config_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = load_curves(data_root, ema_span=cfg.ema_span, eps_T=cfg.eps_T, lr_scale="max")
    _, eval_idx = strict_indices(curves, cfg)
    model = tuned_mtl_model(config_dir)
    metrics, predictions = evaluate_models(curves, eval_idx, [model], cfg)
    manifest = build_manifest(curves, cfg.fit_schedule, cfg.test_schedule)
    write_result_bundle(out_dir, metrics, predictions, [model], None, manifest, curves)
    return metrics, predictions
