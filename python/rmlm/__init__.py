"""rmlm package: two implementations kept side by side on purpose.

- validated_lib.py : our own from-scratch implementation, directly off the
  paper's formulas, using one shared global radius/exceedance-set for every
  max-projection. Thoroughly unit-tested (see python/tests). Correlation with
  ground truth on synthetic data ~0.97-0.98. Use this as the DEFAULT/primary
  implementation.

- original_port.py : line-by-line port of the authors' own R code
  (R/original/CJS_functions_code.R). Uses a LOCAL radius/dimension per
  max-projection (the authors' actual method -- see docs for details).
  In our tests so far it shows more estimation variance (~0.86-0.88
  correlation) on the same synthetic data. Kept as a faithful reference
  and as the basis for one of the research-gap experiments (see
  docs/gap10_local_vs_global_estimator.md).
"""
