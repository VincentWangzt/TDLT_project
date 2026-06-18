# TDLT Task 2: Loss-Curve Prediction Under Learning-Rate Schedules

This repository is the cleaned final-project codebase for Topics in Deep Learning Theory
(TDLT), Task 2. It distills the original artifact pile into a reproducible package,
curated results, and a standalone Beamer slide deck.

## Problem

Given observed LLM pretraining loss curves under a learning-rate schedule (LRS), fit a
parametric loss functional on the cosine schedule and predict the loss curve under WSD.
The course data are in `data/loss_curves/gpt_loss+lrs.pkl` and contain three curves:

- `cosine`: the main fitting schedule;
- `wsd`: the main test schedule;
- `811`: an auxiliary unseen schedule used for sanity/generalization evidence.

The main reported metric is EMA-201 WSD full-region RMSE on `step >= 1000`, evaluated
every 20 recorded steps.

## Methods

The repository exposes:

- MTL: Momentum Law / learning-rate annealing baseline.
- MPL: Multi-Power Law style schedule-response baseline.
- FSL: faithful practical ansatz from Functional Scaling Laws, Appendix B.2.
- KMTL: multi-timescale kernelized MTL.
- FSL-MPL+ small/source and NCPL-lite residual variants.
- Tuned MTL: the pinned final hyperparameter-level MTL result found by WSD-adaptive
  autoresearch. It is replayed from stored parameters and explicitly marked as tuned on
  WSD feedback.

## Environment

Create the environment:

```powershell
conda create -y -n tdlt python=3.10
conda run -n tdlt python -m pip install numpy==1.26.4 scipy==1.11.4 matplotlib==3.8.4 tqdm==4.66.4 scikit-learn==1.4.2 pandas==2.2.2
conda run -n tdlt python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
conda run -n tdlt python -m pip install -e .
```

`pandas` is required because the course file is a pandas pickle.

## Reproduce

Run the full curated pipeline:

```powershell
conda run -n tdlt python scripts/reproduce_all.py --out-dir results
```

Or run individual groups:

```powershell
conda run -n tdlt python scripts/run_baselines.py --out-dir results/baselines
conda run -n tdlt python scripts/run_variants.py --out-dir results/variants
conda run -n tdlt python scripts/run_tuned_mtl.py --out-dir results/tuned_mtl
```

Expected headline WSD EMA full RMSE:

| Method | RMSE | Notes |
| --- | ---: | --- |
| MTL | 0.034569 | Clean strict baseline |
| MPL | 0.044353 | Unified strict comparison harness |
| FSL | 0.039657 | Faithful practical FSL ansatz |
| FSL-MPL+ source + NCPL-lite | 0.036702 | Best small-variant full RMSE among non-tuned variants |
| Tuned MTL | 0.015404 | WSD-adaptive tuned result, not an unbiased WSD test |

## Slides

The final standalone deck source is `slides/tdlt_final_project.tex`. Build with:

```powershell
cd slides
latexmk -xelatex tdlt_final_project.tex
```

The deck includes a placeholder for the GitHub URL and group-member metadata.
