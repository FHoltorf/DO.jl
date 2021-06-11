module DO

using Reexport
using LinearAlgebra

"""
finds orthonormal U and Z such that || UZ' - U0 Z0'|| is minimal (via gradient
descent)
"""
function grad_descent(Z0, U0; μ = 0.1, ϵ = 1e-6, iter_max = 1000)
    iter = 0
    Z = Z0
    U = U0
    while iter < iter_max
        dZ = (Z - Z0*(U0'*U))
        dU = -(I - U*U')*U0*(Z0'*Z*pinv(Z'*Z))
        Z -= μ*dZ
        U -= μ*dU
        orthonormalize!(U, Z; μ = 0.1)
        if √(norm(dZ)^2 + norm(dU)^2) < ϵ
            break
        end
        iter += 1
    end
    return Z, U
end

"""
finds orthonormal ̄U such that || (U-̄U)Z || is minimal (via gradient descent)
"""
function orthonormalize!(U, Z; ϵ = 1e-6, μ = 0.1, iter_max = 1000)
    iter = 0
    K = U'*U
    A = Matrix{Float64}(I, size(K))
    Ainv = A
    dA = A'*K*A - I
    while norm(dA) > ϵ && iter < iter_max
        A += μ*dA
        Ainv -= μ*Ainv*A*Ainv
        dA = - K * A * (A'*K*A - I)
        iter += 1
    end
    U .= U*A
    Z .= Z*Ainv'
end

"""
augments low rank representation with relevant directions from normal cone
"""
function augment_rank(N, U, Z, σ_thresh; μ = 0.1, ϵ = 1e-6, iter_max = 100)
    V, Σ, P = svd(N)
    r = sum(Σ .>= σ_thresh)
    N_U = V[:, 1:r]
    N_Z = P[:, 1:r]*Diagonal(Σ[1:r])
    U0 = cat(U, N_U, dims=2)
    Z0 = cat(Z, N_Z, dims=2)
    return grad_descent(Z0, U0; μ = μ, ϵ = ϵ, iter_max = iter_max)
end

"""
finds normal to low rank matrix manifold
"""
function normal_direction(flux, U,Z)
    return (I - U*U')*flux(U*Z')*(I - Z*pinv(Z'*Z)*Z')
end

end
