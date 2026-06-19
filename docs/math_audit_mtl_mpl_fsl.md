# Math Audit: MTL, MPL, and FSL

Scope: compare `tdlt_project/src/tdlt_losscurves/models/{mtl.py,mpl_like.py,fsl.py}` against the local papers in `DL-project/paper`.

## Paper Equations Checked

- MTL: `SCALING LAW WITH LEARNING RATE ANNEALING.pdf`, Eq. (1) and Eq. (5):
  `L(s)=L0 + A*S1^{-alpha} - C*S2`, `S1=sum_i eta_i`, `m_i=lambda*m_{i-1}+(eta_{i-1}-eta_i)`, `S2=sum_i m_i`.
- MPL: `A multi-power law for loss curve prediction across learning rate schedules.pdf`, Eq. (1)/(2), restated as Eq. (15):
  `L(t)=L0 + A*(S1(t)+SW)^{-alpha} - B*sum_k (eta_{k-1}-eta_k) * (1 - (1 + C*eta_k^{-gamma}*S_k(t))^{-beta})`,
  where `S_k(t)=sum_{tau=k}^t eta_tau`.
- FSL: `Functional Scaling Laws in Kernel Regression.pdf`, Appendix B.2 Eq. (30), then the fixed-model practical form:
  `L(k)=L0+c1/T(k)^s - c2*sum_i (eta_{i-1}-eta_i)*(c3+1/T(i)^s)*(1 - 1/(1+c4*(T(k)-T(i)))^gamma)`.

Note: the relevant Eq. (30) in the local papers is the FSL paper's Appendix B.2 practical FSL ansatz. The MPL paper's core MPL formula is Eq. (1)/(2)/(15), not that Eq. (30).

## Verdicts

### MTL

`mtl.py` is algebraically faithful to the Momentum Law used in the paper:

- `Curve.T` is `S1` via cumulative learning rate.
- `mtl_memory` computes `m_i=lambda*m_{i-1}+delta_i` and returns `cumsum(m)`, exactly `S2`.
- `predict_mtl` applies `L0 + A*T^{-alpha} - C*S2`.

Adaptations:

- In the unified baseline runner, MTL uses peak-normalized learning rate and `eps_T=1.0`, so it is a reparameterized version of the paper formula, not raw `sum eta_i`.
- The fit objective uses EMA-smoothed course loss and a weighted Huber log objective.

### MPL

`mpl_like.py` now matches the MPL paper's inclusive `S_k(t)` indexing under exact-source settings, after the same-step source fix in `response_sum`.

Matches:

- The parameter tuple and outer structure match: `L0 + A*T^{-alpha} - B*response_sum`.
- The nonlinear response matches the MPL `G(x)=1-(1+C*x)^(-beta)` form with `x=eta_k^{-gamma}*S_k(t)`.
- With `source_stride=1`, monotone LR, and no warmup offset, the implemented active mask plus `gap + eta_s` recovers `S_k(t)`, including `S_k(k)=eta_k`.

Remaining adaptations:

- The code uses `delta_pos`, clipping re-warmup/increasing-LR contributions to zero. The paper formula uses signed `(eta_{k-1}-eta_k)` under its non-increasing post-warmup schedule assumption.
- The default unified protocol uses `source_stride=100`, which compresses sources and is an approximation to the full sum.
- MTL/MPL baselines use peak-normalized LR and `eps_T=1.0`; the paper formula is written in raw LR sum plus optional warmup sum `SW`.

Synthetic check for the same-step fix:

```text
eta = [1.0, 0.5, 0.5], C=2.0, beta=0.7, gamma=0.4
code response at t=1: 0.22254478
paper response at t=1: 0.22254478
```

### FSL

`fsl.py` matches the practical FSL Appendix B.2 Eq. (30) after dropping the fixed model-size term, under exact-source settings.

Matches:

- `compute_fsl_features` extracts source pairs `(T(i), eta_{i-1}-eta_i)`.
- `_response` computes `(eta_{i-1}-eta_i)*(c3+T(i)^(-s))*(1-(1+c4*(T(k)-T(i)))^(-gamma))`.
- `predict_fsl_params` computes `L0+c1*T(k)^(-s)-c2*response`.
- The FSL path in `experiments.py` loads raw LR with `eps_T=1e-8`, matching the faithful reproduction convention more closely than the normalized MTL/MPL path.

Adaptations:

- The implementation clips LR increases by using positive LR drops only. This is fine for monotone decay/post-warmup schedules but not a literal signed Eq. (30) for re-warmup cases.
- `source_stride > 1` compresses source events and is an approximation. The generated FSL `fitted_params.json` records the stride used for each fresh run.
- The optimizer is L-BFGS-B over transformed positive coefficients rather than the Adam setup described in the paper appendix. This changes fitting procedure, not the prediction formula.

Numerical sanity checks:

```text
MTL code vs paper recurrence: max abs diff 0.0
FSL code vs Eq. (30), source_stride=1: max abs diff 0.0
FSL source_stride=2 vs exact source sum: max abs diff 0.09043986901376
```

## Bottom Line

- MTL: mathematically faithful modulo LR scaling/epsilon and fitting-protocol adaptations.
- MPL: formula-faithful for the inclusive `S_k(t)` response with `source_stride=1`; practical runs still use source compression plus normalized LR/epsilon protocol adaptations.
- FSL: formula-faithful to Appendix B.2 Eq. (30) with `source_stride=1`; practical runs with source compression are approximations but preserve the intended algebra.
