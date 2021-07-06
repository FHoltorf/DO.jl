"""
    direct time marching refers essentially to explicit Euler integration.
"""
function direct_time_marching!(Δt,U,Z,L)
    dU = (I - U*U')*L*Z*pinv(Z'*Z)
    dZ = L'*U
    Z .+= Δt * dZ
    U .+= Δt * dU
end

function direct_time_marching!(Δt,U,Z,L_U,L_Z)
    dU = (I - U*U')*L_U*(L_Z'*Z*pinv(Z'*Z))
    dZ = L_Z*(L_U'*U)
    U .+= Δt * dU
    Z .+= Δt * dZ
end

"""
    explicit time step with subsequent projection into low rank manifold
"""
step_and_project!(Δt, U, Z, L) = project_to_low_rank!(U, Z, U*Z' + Δt*L)
step_and_project!(Δt, U, Z, L_U, L_Z) = project_to_low_rank!(U, Z, cat(U, L_U, dims=2), cat(Z, Δt*L_Z, dims=2))

"""
    integration along geodesics of the low rank matrix manifold
    --> High order integration schemes
    --> Direct time marching approximates this only to first order
"""
function exponential_map_integration!(Δt, U, Z, dU, dZ)
# TBD
# requires additional implementation of numerical schemes to solve the
# geodesic equations (ODE):
# d²Z/dt² - Z (dU/dt)ᵀ dU/dt = 0
# d²U/dt² + U (dU/dt)ᵀ dU/dt + 2 dU/dt (dZ/dt)ᵀ Z inv(ZᵀZ) = 0
# with IC: dU/dt(0) = dU, U(0) = U
#          dZ/dt(0) = dZ, Z(0) = Z
# The ODE system is solved for [0, Δt] and the update is given by U(Δt), Z(Δt)
# positive: We can write tailored code to integrate these equations
end
