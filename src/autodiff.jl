using ChainRulesCore

@non_differentiable generate_horizontal_rules()
@non_differentiable generate_vertical_rules()
@non_differentiable hamiltonian(model::HamiltonianModel)
@non_differentiable Hubbard_hand(model::HamiltonianModel)
@non_differentiable hop_pair_hand(t,γ)

ChainRulesCore.rrule(::typeof(T_parity_conserving),T::AbstractArray) = T_parity_conserving(T), dT -> (NoTangent(), T_parity_conserving(dT))

ChainRulesCore.rrule(::typeof(t2Z), A::AbstractArray) = t2Z(A), dAZ2 -> (NoTangent(), Z2t(dAZ2))

ChainRulesCore.rrule(::typeof(particle_conserving),T::AbstractArray) = particle_conserving(T), dT -> (NoTangent(), particle_conserving(dT))
