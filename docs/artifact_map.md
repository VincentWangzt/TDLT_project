# Artifact Map

This document records how the source pile was distilled into this clean repository.

## Original Artifact Families

- `final_project/Reproduction/MTL/tdlt_task2_original_compat_20260525_021411`
  contains the original-compatible MTL adapter. It verifies the Momentum Law formula and
  course-data conversion against the earlier clean wrapper.
- `final_project/Reproduction/MPL/MultiPowerLaw/tdlt_task2_original_compat_20260525_021411`
  contains the original-compatible MPL adapter. Direct original `main.py` was avoided
  because the upstream loader reconstructs learning-rate schedules from hard-coded paper
  filenames instead of using the course `lr` column.
- `final_project/Improvement/tdlt_task2_improvement_20260527_004532`
  is the unified comparison harness for MTL, MPL, FSL-MPL+, KMTL, NCPL-lite, metrics,
  figures, and auxiliary protocols.
- `DL-project/Reproduction/FSL/tdlt_task2_fsl_repro_20260614_050854`
  is the faithful practical FSL reproduction based on Appendix B.2 Eq. (30).
- `DL-project/Improvement/autoresearch/loss-curve-20260608`
  contains the autoresearch trace. This final repo keeps only curated metadata for the
  selected tuned-MTL search setting, not loadable winning parameters or the raw
  296-attempt search loop.

## Final Selection Rules

- Ship one coherent package instead of nested timestamped folders.
- Preserve the course pickle, the final slide template assets, curated result CSVs,
  and enough provenance to explain where each method came from.
- Do not ship raw search journals or generated LaTeX build artifacts.
- Mark tuned MTL as WSD-adaptive model selection, while using 8-1-1 as auxiliary evidence.
- Generate all submitted fitted parameters from the repo's fitting logic with fixed seeds.

## Course Data

`data/loss_curves/gpt_loss+lrs.pkl` contains three pandas DataFrames with columns:

- `step`
- `Metrics/loss`
- `lr`

The Chinese readme says to read the pickle with pandas and explains that the 8-1-1 schedule
decays at 80% and 90% by a factor of `1/sqrt(10)`.
