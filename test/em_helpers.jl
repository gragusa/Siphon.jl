_em_asmask(f) = f isa AbstractVector ? Matrix(Diagonal(collect(Bool, f))) : f
_em_asmat(x) = x isa AbstractVector ? Matrix(Diagonal(float.(x))) : float.(x)

# EM on explicit system matrices. Sets the masks of an EMWorkspace and runs
# Siphon.em_estimate!. `H`, `Q`, `H_free` and `Q_free` accept a matrix or a
# vector of diagonal entries; for a vector mask the other cells stay fixed.
function em_matrices(Z, T, R, H, Q, y, a1, P1;
        Z_free = trues(size(Z)), T_free = trues(size(T)),
        H_free = trues(size(_em_asmat(H))), Q_free = trues(size(_em_asmat(Q))),
        T_ties = Vector{CartesianIndex{2}}[], maxiter = 500, tol = 1e-6,
        warn = false, kwargs...)
    kf_ws = KalmanWorkspace(float.(Z), _em_asmat(H), float.(T), float.(R), _em_asmat(Q),
        float.(a1), float.(P1), size(y, 2))
    em_ws = Siphon.EMWorkspace(kf_ws)
    em_ws.Z_free .= Z_free
    em_ws.H_free .= _em_asmask(H_free)
    em_ws.H_diag_only = !any(I -> I[1] != I[2] && em_ws.H_free[I],
        CartesianIndices(em_ws.H_free))
    em_ws.T_free .= T_free
    em_ws.Q_free .= _em_asmask(Q_free)
    append!(em_ws.T_ties, T_ties)
    return Siphon.em_estimate!(kf_ws, em_ws, y; maxiter, tol, warn, kwargs...)
end
