# TDLT Task 2: Loss-Curve Prediction Under Learning-Rate Schedules

This repository contains the final-project implementation and Beamer presentation for
Topics in Deep Learning Theory (TDLT), Task 2. The project studies how a loss functional
fitted on one learning-rate schedule transfers to schedules with different shapes.

Repository: [VincentWangzt/TDLT_project](https://github.com/VincentWangzt/TDLT_project)

## Problem

The main protocol fits a parametric loss functional on the cosine loss curve and predicts
the WSD loss curve. The course data are stored in
**data/loss_curves/gpt_loss+lrs.pkl** and contain three schedules:

- **cosine**, used for parameter fitting;
- **wsd**, used for the main evaluation;
- **811**, a three-stage auxiliary schedule evaluated with the fitted parameters.

The 8-1-1 schedule reduces the learning rate at 80% and 90% of training. Its three
levels are eta, eta divided by the square root of 10, and eta divided by 10.

## Methods

### Published baselines

- **MTL** implements the learning-rate annealing law of Tissue, Wang, and Wang,
  [Scaling Law with Learning Rate Annealing](https://arxiv.org/abs/2408.11029).
- **MPL** implements the multi-power schedule-response model of Luo et al.,
  [A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules](https://arxiv.org/abs/2503.12811).
- **FSL** implements the practical fixed-model-size ansatz from Li et al.,
  [Functional Scaling Laws in Kernel Regression](https://arxiv.org/abs/2509.19189),
  Appendix B.2.

### Main contribution: Tuned-MTL

Tuned-MTL retains the complete MTL prediction formula and expands its estimation
procedure in three ways:

- the memory grid contains 13 values, with finer resolution near one;
- selected points in the final 20% of the cosine fitting region receive weight 3
  instead of the baseline weight 2;
- 50 random restarts explore the expanded grid instead of the baseline 20.

The selected memory value is lambda = 0.9997. Its approximate exponential half-life
is 2310 recorded steps, compared with 69 steps for the baseline selection
lambda = 0.99. This long-memory setting produces the strongest full-region WSD result.

The exact Tuned-MTL configuration is recorded in
**configs/tuned_mtl_selection.json**. Fitted parameters are regenerated from the cosine
curve with the fixed seed stored in that configuration.

### Exploratory variants

- **Two-scale and three-scale MTL** mix fixed exponential memory kernels.
- **Compact FSL-MPL** uses a simplified intrinsic-time power response.
- **Source-weighted FSL-MPL** varies response strength with source time.
- **Residual-corrected FSL-MPL** adds a regularized correction fitted to cosine
  residuals.

## Experimental protocol

The implementation applies the following protocol consistently:

| Component | Setting |
| --- | --- |
| Prediction target | Exponentially weighted moving average with span 201 |
| Evaluation start | Step 1000 |
| Main fitting schedule | Cosine |
| Main evaluation schedule | WSD |
| Auxiliary schedule | 8-1-1 |
| Fit sampling | Every 800 steps |
| Additional tail sampling | Every 250 steps in the final 20% |
| Evaluation sampling | Every 20 steps, plus the final observation |
| Baseline MTL/MPL tail weight | 2 |
| Tuned-MTL tail weight | 3 |
| Baseline MTL/MPL restarts | 20 |
| FSL restarts | 10 |
| Tuned-MTL restarts | 50 |

The exponentially weighted target is computed recursively with smoothing coefficient
2 divided by 202. Model fitting uses Huber loss on logarithmic residuals with threshold
0.001. The headline metric is full-region RMSE against the smoothed WSD curve.

MTL and MPL operate on learning rates normalized by the peak rate of each schedule.
The FSL reproduction uses the original learning-rate scale and its own intrinsic-time
offset, following the practical implementation.

## Implementation adaptation

The published formulations are connected to the course data through a shared adapter.
The adapter reads the provided training step, loss, and learning-rate columns directly
and leaves the original pickle unchanged.

The reference MPL data loader reconstructs schedules from paper-specific filenames.
Our adapter supplies the course learning-rate observations to the same mathematical
evaluator. Formula-level checks compare the adapted MTL and MPL evaluators with their
reference-compatible computations under identical parameters. FSL includes independent
sanity checks for zero response under constant learning rate and monotone response with
intrinsic-time separation.

A shared evaluation layer applies the same target construction, index selection, region
definitions, and metrics to every method.

## Generated artifacts

Each experiment directory contains:

- **predictions.csv**, with the evaluated steps, observations, predictions, and signed
  errors;
- **metrics.csv**, with full and regional metrics for smoothed and raw loss;
- **fitted_params.json**, with fitted parameters and the fitting signal;
- **restart_summary.csv**, with optimizer outcomes for each restart;
- **data_manifest.csv**, with schedule names, row counts, step ranges, learning-rate
  statistics, and final losses;
- **figures/**, with schedule, prediction, error, and summary plots.

The combined tables **results/all_predictions.csv** and **results/all_metrics.csv**
support the presentation. RMSE values are recomputed from the saved prediction rows in
the automated checks.

## Environment

Create the environment:

~~~powershell
conda create -y -n tdlt python=3.10
conda run -n tdlt python -m pip install numpy==1.26.4 scipy==1.11.4 matplotlib==3.8.4 tqdm==4.66.4 scikit-learn==1.4.2 pandas==2.2.2
conda run -n tdlt python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
conda run -n tdlt python -m pip install -e .
~~~

Pandas is required because the course artifact is a pandas pickle. Exact environment
specifications are also available in **environment.yml**, **requirements.txt**, and
**pyproject.toml**.

## Reproduce

Run the complete curated pipeline:

~~~powershell
conda run -n tdlt python scripts/reproduce_all.py --out-dir results
~~~

Run individual experiment groups:

~~~powershell
conda run -n tdlt python scripts/run_baselines.py --out-dir results/baselines
conda run -n tdlt python scripts/run_variants.py --out-dir results/variants
conda run -n tdlt python scripts/run_tuned_mtl.py --out-dir results/tuned_mtl
~~~

Run the automated checks:

~~~powershell
python -m pytest
~~~

## Headline results

WSD metrics below compare predictions with the EMA-smoothed full curve from step 1000,
evaluated every 20 steps and at the final observation.

| Method | RMSE | MAE | R-squared | Final absolute error |
| --- | ---: | ---: | ---: | ---: |
| MTL | 0.034569 | 0.028816 | 0.963640 | **0.004317** |
| MPL | 0.040219 | 0.031148 | 0.950783 | 0.029069 |
| FSL | 0.040358 | 0.035311 | 0.950443 | 0.053288 |
| **Tuned-MTL** | **0.015461** | **0.010069** | **0.992727** | 0.016741 |
| Residual-corrected FSL-MPL | 0.038326 | 0.027214 | 0.955309 | 0.081484 |

Tuned-MTL reduces full-region WSD RMSE by 55.3% relative to baseline MTL. Baseline MTL
retains the lowest final absolute error.

## Slides

The presentation source is **slides/tdlt_final_project.tex**.

~~~powershell
cd slides
latexmk -xelatex tdlt_final_project.tex
~~~
