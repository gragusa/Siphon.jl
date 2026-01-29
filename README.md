# Siphon.jl

[![CI](https://github.com/gragusa/Siphon.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/gragusa/Siphon.jl/actions/workflows/ci.yml) [![codecov.io](http://codecov.io/github/gragusa/Siphon.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/Siphon.jl?branch=master) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) ![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)

Linear State Space Models in Julia: Kalman filtering, smoothing, and estimation.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/Siphon.jl")
```

## Features

- AD-compatible Kalman filter/smoother
- Parameters estimation via EM algorithm or numerical optimization
- Missing data handling
- DSL for model specification with pre-built templates 

## Documentation

See `docs/` for full documentation and `examples/` for worked examples.

## License

MIT
