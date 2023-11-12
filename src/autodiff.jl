using ChainRulesCore

@non_differentiable initial_consts(key)

# ChainRulesCore.rrule(::typeof(T_parity_conserving),T::AbstractArray) = T_parity_conserving(T), dT -> (NoTangent(), T_parity_conserving(dT))

# ChainRulesCore.rrule(::typeof(t2Z), A::AbstractArray) = t2Z(A), dAZ2 -> (NoTangent(), Z2t(dAZ2))

# ChainRulesCore.rrule(::typeof(particle_conserving),T::AbstractArray) = particle_conserving(T), dT -> (NoTangent(), particle_conserving(dT))
