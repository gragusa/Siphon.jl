# Matrix Helpers

This page documents helper functions for constructing state space matrices.

## Diagonal Matrices

```@docs
Siphon.diag_free
Siphon.scalar_free
```

Fixed matrices are plain Julia matrices (`zeros`, `ones`, `Diagonal`); `custom_ssm`
also accepts a scaled identity `I`.

## Covariance Matrices

```@docs
Siphon.cov_free
Siphon.CovFree
```

## Structured Matrices

```@docs
Siphon.lower_triangular_free
Siphon.symmetric_free
```

## Selection and Companion Matrices

```@docs
Siphon.selection_mat
Siphon.companion_free
```

## Combining Blocks

```@docs
Siphon.block_diag
```
