module DO

using LinearAlgebra

export integrate, DOProblem

include("utils.jl")
include("low_rank_timestepping.jl")
mutable struct DOProblem
    flux # semidiscretization
    factorized # does the flux factorize? true/false
    U0 # initial set of modes n × s
    Z0 # initial set of stochastic coefficients l × s
end

function integrate(prob::DOProblem, ts, σ_min, integration_method = :direct_time_marching)
    U, Z = prob.U0, prob.Z0
    Us, Zs = [U], [Z]
    integrator = getfield(DO, Symbol(string(integration_method,:!)))
    for i in 2:length(ts)
        println(i)
        Δt = ts[i] - ts[i-1]
        if prob.factorized
            L_U, L_Z = prob.flux(U,Z,ts[i])
            N = Δt*normal_direction(U, Z, L_U, L_Z)
            if norm(N) > σ_min
                U, Z = augment_rank(N, U, Z, σ_min)
                println("   rank augmented")
            else
                integrator(Δt, U, Z, L_U, L_Z)
            end
        else
            L = prob.flux(U*Z', ts[i])
            N = Δt*normal_direction(U, Z, L)
            if norm(N) > σ_min
                U, Z = augment_rank(N, U, Z, σ_min)
                println("   rank augmented!")
            else
                integrator(Δt, U, Z, L)
            end
        end
        push!(Us, U)
        push!(Zs, Z)
    end
    return Us, Zs
end

end
