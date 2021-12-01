using ChainRulesCore

@non_differentiable generate_horizontal_rules()
@non_differentiable generate_vertical_rules()
@non_differentiable hamiltonian(model::HamiltonianModel)
@non_differentiable swapgate(n1,n2)
@non_differentiable Hubbard_hand(model::HamiltonianModel)
@non_differentiable hop_pair_hand(t,γ)

function ChainRulesCore.rrule(::typeof(parity_conserving),T::Union{Array,CuArray})
	result = parity_conserving(T)
	function pullback_parity_conserving(ΔT)
		return (NoTangent(),parity_conserving(ΔT))
	end
	return result,pullback_parity_conserving
end