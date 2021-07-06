"""
standard Armijo linesearch to determine step length
default parameters c = τ = 0.5 as in Armijo's paper (1966)
"""
function backtracking_linesearch(X, U, Z, dZ, dU, μ0 = 1, c = 0.5, τ = 0.5)
    μ = μ0
    val = norm(X - U*Z')^2
    m = -(norm(dU)^2 + norm(dZ)^2)
    A = X - U*Z'
    a = [dU*Z', U*dZ', dU*dZ']
    while norm(A + μ*(a[1] + a[2]) - μ^2*a[3])^2 > val + μ*c*m
        μ *= τ
    end
    return μ
end

"""
augments low rank representation with relevant directions from normal cone
"""
function augment_rank(N, U0, Z0, σ)
    V, Σ, P = svd(N)
    r = sum(Σ .>= σ)
    N_U = V[:, 1:r]
    N_Z = P[:, 1:r]*Diagonal(Σ[1:r])
    U = cat(U0, N_U, dims=2)
    Z = cat(Z0, N_Z, dims=2)
    project_to_low_rank!(U, Z, U0, Z0)
    return U, Z
end

"""
finds normal to low rank matrix manifold
"""
normal_direction(U, Z, L_U, L_Z) = (I - U*U')*L_U*(L_Z*(I - Z*pinv(Z'*Z)*Z'))
normal_direction(U, Z, L) = (I - U*U')*L*(I - Z*pinv(Z'*Z)*Z')

"""
finds orthonormal ̄U such that || (U-̄U)Z || is minimal (via gradient descent)
"""
function orthonormalize!(U, Z; ϵ = 1e-8, μ = 0.1, iter_max = 1000)
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

    if iter == iter_max
        @warn "Maximum number of iterations exceeded in orthonormalization step"
    end

    U .= U*A
    Z .= Z*Ainv'
end


"""
finds orthonormal U and Z such that || UZ' - U0 Z0'|| is minimal (via gradient
descent)
"""
function project_to_low_rank!(U, Z, U0, Z0; μ = 0.1, ϵ = 1e-8, iter_max = 1000)
    iter = 0
    while iter < iter_max
        dZ = (Z - Z0*(U0'*U))
        dU = -(I - U*U')*U0*(Z0'*Z*pinv(Z'*Z)) # probably better to use a backsolve
        Z .-= μ*dZ
        U .-= μ*dU
        orthonormalize!(U, Z; μ = 0.1)
        if √(norm(dZ)^2 + norm(dU)^2) < ϵ
            break
        end
        iter += 1
    end
    if iter == iter_max
        @warn "Maximum number of iterations exceeded in projection step"
    end
end

"""
finds orthonormal U and Z such that || X - U Z'|| is minimal (via gradient
descent)
"""
function project_to_low_rank!(U, Z, X; μ = 0.1, ϵ = 1e-8, iter_max = 1000)
    iter = 0
    while iter < iter_max
        dZ = (Z - X*U)
        dU = -(I - U*U')*X*Z*pinv(Z'*Z) # probably better to use a backsolve
        Z .-= μ*dZ
        U .-= μ*dU
        orthonormalize!(U, Z; μ = 0.1)
        if √(norm(dZ)^2 + norm(dU)^2) < ϵ
            break
        end
        iter += 1
    end

    if iter == iter_max
        @warn "Maximum number of iterations exceeded in projection step"
    end
end
