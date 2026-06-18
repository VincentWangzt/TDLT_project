# Methodological Note On The Tuned MTL Result

The tuned MTL result is the strongest numerical WSD result in the artifacts:

- WSD EMA full RMSE: `0.015403891057018722`
- MTL lambda: `0.9997`
- Tail fitting weight: `3.0`
- Wider lambda grid than the frozen MTL baseline

This is not direct fitting on WSD loss values. However, the autoresearch process repeatedly
evaluated WSD RMSE and used it to keep or discard attempts. Therefore WSD served as an
adaptive development signal. The tuned result should be described as a WSD-adaptive
development result, not as an unbiased held-out test estimate.

The cleanest baseline comparison remains the strict cosine-fit to WSD-test table for MTL,
MPL, and faithful FSL. The tuned result is still valuable because it shows that the original
MTL functional form was under-optimized: the winning change is a hyperparameter/optimization
change, not a new model family.
