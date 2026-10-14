#=
show.jl

Display, value equality and hashing for the user-facing types.

The two-argument `show` methods print one line. Those of small value types
(`SSMParameter`, `FreeParam` and the priors) are valid constructor calls; the
others summarize dimensions and run status. The `text/plain` methods of specs,
parameter sets and models add a few lines of detail.
=#

_sig(x) = round(x; sigdigits = 8)

_plural(k, word, words = word * "s") = k == 1 ? "1 $word" : "$k $words"

# ============================================
# Parameters
# ============================================

function Base.show(io::IO, prm::SSMParameter{Float64})
    print(io, "SSMParameter(")
    show(io, prm.name)
    kw = String[]
    isfinite(prm.lower) && push!(kw, "lower = $(repr(prm.lower))")
    isfinite(prm.upper) && push!(kw, "upper = $(repr(prm.upper))")
    # The keyword constructor recomputes a default init from the bounds; any other
    # init is printed, with `default_init = true` if it is a default all the same.
    if !(prm.default_init && isequal(prm.init, _default_init(prm.lower, prm.upper)))
        push!(kw, "init = $(repr(prm.init))")
        prm.default_init && push!(kw, "default_init = true")
    end
    isempty(kw) || print(io, "; ", join(kw, ", "))
    print(io, ")")
end

function Base.show(io::IO, prm::SSMParameter{T}) where {T}
    print(io, "SSMParameter{", T, "}(")
    join(
        io, (repr(prm.name), repr(prm.lower), repr(prm.upper), repr(prm.init),
            repr(prm.default_init)), ", ")
    print(io, ")")
end

function Base.show(io::IO, fp::FreeParam)
    print(io, "FreeParam(")
    fp.name === nothing || show(io, fp.name)
    kw = String[]
    for f in (:lower, :upper, :init)
        v = getfield(fp, f)
        v === nothing || push!(kw, "$f = $(repr(v))")
    end
    if !isempty(kw)
        fp.name === nothing || print(io, "; ")
        join(io, kw, ", ")
    end
    print(io, ")")
end

# ============================================
# Matrix specifications
# ============================================

_cell_string(e::FixedValue) = repr(e.value)
_cell_string(e::ParameterRef) = string(e.name)

_cell(ms::SSMMatrixSpec, i, j) = get(ms.elements, (i, j), ms.default)

# Names of the parameters in the cells of `ms`, in column-major order.
function _param_refs(ms::SSMMatrixSpec)
    names = Symbol[]
    for j in 1:ms.dims[2], i in 1:ms.dims[1]

        e = _cell(ms, i, j)
        e isa ParameterRef && e.name ∉ names && push!(names, e.name)
    end
    return names
end

_param_refs(e::MatrixExpr) = [prm.name for prm in e.params]
_param_refs(e::CovMatrixExpr) = vcat(e.σ_param_names, e.corr_param_names)
_param_refs(e::MeasurementExpr) = [prm.name for prm in e.params]
_param_refs(::Any) = Symbol[]

function Base.show(io::IO, ms::SSMMatrixSpec)
    refs = _param_refs(ms)
    print(io, "SSMMatrixSpec(", ms.dims[1], "×", ms.dims[2], ", ")
    isempty(refs) ? print(io, "fixed") : print(io, "free: ", join(refs, ", "))
    print(io, ")")
end

# Cells are printed in a grid when the matrix is small enough to read that way.
function Base.show(io::IO, ::MIME"text/plain", ms::SSMMatrixSpec)
    rows, cols = ms.dims
    print(io, rows, "×", cols, " SSMMatrixSpec")
    if rows * cols > 400
        refs = _param_refs(ms)
        print(io, isempty(refs) ? " (fixed)" : " (free: $(join(refs, ", ")))")
        return
    end
    print(io, ":")
    cells = [_cell_string(_cell(ms, i, j)) for i in 1:rows, j in 1:cols]
    widths = [maximum(textwidth, view(cells, :, j)) for j in 1:cols]
    for i in 1:rows
        print(io, "\n")
        for j in 1:cols
            print(io, "  ", rpad(cells[i, j], widths[j]))
        end
    end
end

function Base.show(io::IO, e::MatrixExpr)
    print(io, "MatrixExpr(", e.dims[1], "×", e.dims[2], ", free: ",
        join(_param_refs(e), ", "), ")")
end

function Base.show(io::IO, e::CovMatrixExpr)
    print(io, "CovMatrixExpr(", e.n, "×", e.n, ", ",
        _plural(length(e.σ_param_names), "standard deviation"), ", ",
        _plural(length(e.corr_param_names), "correlation"), ")")
end

function Base.show(io::IO, e::MeasurementExpr)
    print(io, "MeasurementExpr(free: ", join(_param_refs(e), ", "), ")")
end

# ============================================
# Model specifications
# ============================================

function _initialization(spec::SSMSpec)
    uses_exact_diffuse(spec) ? "exact diffuse" :
    "approximate diffuse"
end
_initialization(::EKFSpec) = "approximate diffuse"

function _spec_header(io::IO, spec::AbstractSSMSpec)
    print(io, nameof(typeof(spec)), "(", repr(spec.name), ", m = ", spec.n_states,
        ", p = ", spec.n_obs, ", r = ", spec.n_shocks, ", ",
        _plural(length(spec.params), "parameter"))
end

function Base.show(io::IO, spec::AbstractSSMSpec)
    _spec_header(io, spec)
    print(io, ")")
end

# The parameters used by each part of the system: `[(:Z, [:λ]), (:H, [...]), ...]`.
function _matrix_params(spec::SSMSpec)
    parts = [(:Z, spec.Z), (:H, spec.H), (:T, spec.T), (:R, spec.R), (:Q, spec.Q),
        (:P1, spec.P1)]
    out = [(k, _param_refs(get(spec.matrix_exprs, k, ms))) for (k, ms) in parts]
    a1 = unique(e.name for e in spec.a1 if e isa ParameterRef)
    insert!(out, 6, (:a1, collect(Symbol, a1)))
    spec.P1_inf === nothing || push!(out, (:P1_inf, _param_refs(spec.P1_inf)))
    return out
end

function _matrix_params(spec::EKFSpec)
    parts = [(:H, spec.H), (:T, spec.T), (:R, spec.R), (:Q, spec.Q), (:P1, spec.P1)]
    out = [(k, _param_refs(get(spec.matrix_exprs, k, ms))) for (k, ms) in parts]
    pushfirst!(out, (:measurement, _param_refs(spec.measurement)))
    a1 = unique(e.name for e in spec.a1 if e isa ParameterRef)
    insert!(out, 6, (:a1, collect(Symbol, a1)))
    return out
end

function Base.show(io::IO, ::MIME"text/plain", spec::AbstractSSMSpec)
    print(io, nameof(typeof(spec)), " ", spec.name, " with ",
        _plural(spec.n_states, "state"), ", ", _plural(spec.n_obs, "observed series", "observed series"),
        ", ", _plural(spec.n_shocks, "shock"), ", ", _initialization(spec),
        " initialization")
    params = spec.params
    print(io, "\n", _plural(length(params), "parameter"), isempty(params) ? "" : ":")
    w = isempty(params) ? 0 : maximum(prm -> textwidth(string(prm.name)), params)
    for prm in params
        print(io, "\n  ", rpad(string(prm.name), w), "  ∈ (", prm.lower, ", ", prm.upper,
            "), init ", prm.init, prm.default_init ? " (default)" : "")
    end
    free = [(k, refs) for (k, refs) in _matrix_params(spec) if !isempty(refs)]
    isempty(free) && return
    print(io, "\nFree cells:")
    w = maximum(((k, _),) -> textwidth(string(k)), free)
    for (k, refs) in free
        print(io, "\n  ", rpad(string(k), w), "  ", join(refs, ", "))
    end
end

# ============================================
# System matrices
# ============================================

function _dims_string(p::KFParms)
    p.Z isa Real ? "scalar" :
    "p = $(size(p.Z, 1)), m = $(size(p.Z, 2)), r = $(size(p.Q, 1))"
end

function _dims_string(p::EKFParms)
    p.T isa Real ? "scalar" :
    "p = $(size(p.H, 1)), m = $(size(p.T, 1)), r = $(size(p.Q, 1))"
end

Base.show(io::IO, p::KFParms) = print(io, "KFParms(", _dims_string(p), ")")

function Base.show(io::IO, p::EKFParms)
    print(io, "EKFParms(", _dims_string(p), ", ", nameof(typeof(p.measurement)), ", ")
    show(io, p.jacobian_mode)
    print(io, ")")
end

function _show_system(io::IO, p, fields)
    show(io, p)
    ctx = IOContext(io, :limit => true)
    for f in fields
        print(io, "\n", f, ": ")
        x = getfield(p, f)
        x isa AbstractArray ? show(ctx, MIME"text/plain"(), x) : show(ctx, x)
    end
end

function Base.show(io::IO, ::MIME"text/plain", p::KFParms)
    _show_system(io, p, (:Z, :H, :T, :R, :Q))
end
Base.show(io::IO, ::MIME"text/plain", p::EKFParms) = _show_system(io, p, (:H, :T, :R, :Q))

# ============================================
# Filter and smoother output
# ============================================

# "loglik = …", "not run", or the period in which the filter stopped.
function _run_string(x::_FilterOutput)
    status = _run_status(x)
    status == 0 && return "loglik = $(_sig(_filter_store(x).loglik))"
    status == _NOT_RUN && return "not run"
    return "failed in period $status"
end

function Base.show(io::IO, r::Union{KalmanFilterResult, EKFFilterResult})
    print(io, nameof(typeof(r)), "(p = ", size(r.vt, 1), ", m = ", size(r.at, 1),
        ", n = ", size(r.at, 2), ", ", _run_string(r), ")")
end

function Base.show(io::IO, r::DiffuseFilterResult)
    print(io, "DiffuseFilterResult(p = ", size(r.vt, 1), ", m = ", size(r.at, 1),
        ", n = ", size(r.at, 2), ", d = ", r.d, ", ", _run_string(r), ")")
end

function Base.show(io::IO, r::SmootherResult)
    print(io, "SmootherResult(m = ", size(r.αs, 1), ", n = ", size(r.αs, 2),
        r.Pcross === nothing ? "" : ", with cross-lag covariances", ")")
end

function Base.show(io::IO, r::DisturbanceSmootherResult)
    print(io, "DisturbanceSmootherResult(p = ", size(r.εs, 1), ", r = ", size(r.ηs, 1),
        ", n = ", size(r.εs, 2), ")")
end

function Base.show(io::IO, r::ForecastResult)
    print(io, "ForecastResult(p = ", size(r.yhat, 1), ", m = ", size(r.a, 1),
        ", h = ", size(r.yhat, 2), ", origin = ", r.origin, ")")
end

function _workspace_dims(io::IO, name, ws, T)
    print(io, name, "{", T, "}(p = ", ws.obs_dim, ", m = ", ws.state_dim,
        ", r = ", ws.shock_dim, ", n = ", ws.n_times)
end

function Base.show(io::IO, ws::KalmanWorkspace{T}) where {T}
    _workspace_dims(io, nameof(typeof(ws)), ws, T)
    print(io, ", ", _run_string(ws), ws.smoothed && ws.status == 0 ? ", smoothed" : "", ")")
end

function Base.show(io::IO, ws::DiffuseKalmanWorkspace{T}) where {T}
    _workspace_dims(io, :DiffuseKalmanWorkspace, ws.base, T)
    ws.base.status == 0 && print(io, ", d = ", ws.d)
    print(io, ", ", _run_string(ws), ")")
end

function Base.show(io::IO, ws::EKFWorkspace{T}) where {T}
    _workspace_dims(io, nameof(typeof(ws)), ws, T)
    print(io, ", ", nameof(typeof(ws.measurement)), ", ")
    show(io, ws.jacobian_mode)
    print(io, ", ", _run_string(ws), ")")
end

function Base.show(io::IO, ws::Union{
        EMWorkspace{T}, SimulationSmootherWorkspace{T}}) where {T}
    (_workspace_dims(io, nameof(typeof(ws)), ws, T); print(io, ")"))
end

function Base.show(io::IO, ws::SmootherWorkspace{T}) where {T}
    print(io, "SmootherWorkspace{", T, "}(m = ", size(ws.alpha, 1), ", n = ",
        size(ws.alpha, 2), ")")
end

# ============================================
# Models
# ============================================

function _fit_string(model::AbstractStateSpaceModel)
    model.fitted || return "not fitted"
    return model.converged ? "fitted" : "fitted, not converged"
end

function Base.show(io::IO, model::StateSpaceModel{T}) where {T}
    print(io, "StateSpaceModel{", T, "}(", repr(model.spec.name), ", n = ",
        model.n_times, ", ", _fit_string(model))
    model.filter_status == 0 && print(io, ", loglik = ", _sig(model.loglik))
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", model::StateSpaceModel)
    show(io, model)
    spec = model.spec
    print(io, "\n", _plural(spec.n_states, "state"), ", ",
        _plural(spec.n_obs, "observed series", "observed series"), ", ", _plural(spec.n_shocks, "shock"))
    if model.fitted && model.iterations > 0
        print(io, "; ", _plural(model.iterations, "EM iteration"))
    end
    params = spec.params
    isempty(params) && return
    if !model.theta_fitted
        print(io, "\nParameters not set")
        return
    end
    print(io, "\nParameters:")
    w = maximum(prm -> textwidth(string(prm.name)), params)
    for (prm, v) in zip(params, model.theta_values)
        print(io, "\n  ", rpad(string(prm.name), w), " = ", _sig(v))
    end
end

function Base.show(io::IO, model::DynamicFactorModel{T}) where {T}
    s = model.spec
    print(io, "DynamicFactorModel{", T, "}(N = ", s.n_obs, ", k = ", s.n_factors,
        ", n = ", model.kf_ws.n_times, ", ", _fit_string(model))
    model.fitted && print(io, ", loglik = ", _sig(model.loglik))
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", model::DynamicFactorModel)
    show(io, model)
    s = model.spec
    print(io, "\nloading_lags = ", s.loading_lags, ", factor_lags = ", s.factor_lags,
        ", error_lags = ", s.error_lags)
    model.fitted && print(io, "; ", _plural(model.iterations, "EM iteration"))
end

# ============================================
# Priors and log densities
# ============================================

Base.show(io::IO, p::NormalPrior) = print(io, "NormalPrior(", p.μ, ", ", p.σ, ")")
function Base.show(io::IO, p::InverseGammaPrior)
    print(io, "InverseGammaPrior(", p.α, ", ", p.β, ")")
end

function Base.show(io::IO, p::ParameterPriors)
    print(io, "ParameterPriors(")
    join(io, ("$k = $(repr(v; context = io))" for (k, v) in pairs(p.priors)), ", ")
    print(io, ")")
end

function Base.show(io::IO, p::CompositePrior)
    print(io, "CompositePrior(")
    join(io, (repr(q; context = io) for q in p.priors), ", ")
    print(io, ")")
end

function Base.show(io::IO, ld::SSMLogDensity)
    print(io, "SSMLogDensity(", repr(ld.spec.name), ", ",
        _plural(length(ld.spec.params), "parameter"), ", ", size(ld.y, 1), "×",
        size(ld.y, 2), " data, prior = ")
    show(io, ld.prior)
    print(io, ")")
end

# ============================================
# Value equality and hashing
# ============================================

# Specs, system matrices and matrix expressions are immutable values whose fields
# include arrays and dictionaries, so the default `==` (`===`) compares those fields
# by identity. These compare and hash field by field instead. The hash salt is the
# type name without parameters, so a `KFParms` of `SMatrix` and one of `Matrix`
# holding the same values are equal and hash alike.
for S in (:SSMMatrixSpec, :SSMSpec, :EKFSpec, :MatrixExpr, :CovMatrixExpr,
    :MeasurementExpr, :ParamExpr, :KFParms, :EKFParms)
    @eval begin
        function Base.:(==)(a::$S, b::$S)
            return all(f -> getfield(a, f) == getfield(b, f), fieldnames(typeof(a)))
        end
        function Base.isequal(a::$S, b::$S)
            return all(f -> isequal(getfield(a, f), getfield(b, f)), fieldnames(typeof(a)))
        end
        function Base.hash(x::$S, h::UInt)
            h = hash($(QuoteNode(S)), h)
            for f in fieldnames(typeof(x))
                h = hash(getfield(x, f), h)
            end
            return h
        end
    end
end
