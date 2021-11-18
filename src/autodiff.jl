using ChainRulesCore

@non_differentiable generate_horizontal_rules()
@non_differentiable generate_vertical_rules()
@non_differentiable hamiltonian(model)
@non_differentiable swapgate(n1,n2)

function ChainRulesCore.rrule(::typeof(parity_conserving),T::Array)
	result = parity_conserving(T)
	function pullback_parity_conservingy(ΔT)
		s = size(ΔT)
		@assert prod(size(ΔT))%2 == 0
		ΔT = reshape(ΔT,[2 for i = 1:Int(log2(prod(s)))]...)
		p = zeros(size(ΔT))
		for index in CartesianIndices(ΔT)
			if mod(sum([i for i in Tuple(index)].-1),2) == 0
				p[index] = 1.0
			end
		end
		return (NoTangent(),reshape(p.*ΔT,size(T)...)) # 
	end
	return result,pullback_parity_conservingy
end