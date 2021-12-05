using ITensors
using OMEinsum

const Cup = [0.0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
const Cdn = [0.0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]
const Nup = [0.0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1]
const Ndn = [0.0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1]
abstract type HamiltonianModel end

@doc raw"
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"
function hamiltonian end

@doc raw"
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"
function hamiltonian end

@doc raw"
    Hubbard(t::Real,U::Real,μ::Real)

return a struct representing Hubbard model
"
struct Hubbard{T<:Real} <: HamiltonianModel
    t::T
	U::T
    μ::T
end

"""
	hamiltonian(model::Hubbard)
"""
function hamiltonian(model::Hubbard)
	t = model.t
	U = model.U
    μ = model.μ
    ampo = AutoMPO()
    sites = siteinds("Electron",2)
    ampo .+= -t, "Cdagup",1,"Cup",2
    ampo .+= -t, "Cdagup",2,"Cup",1
    ampo .+= -t, "Cdagdn",1,"Cdn",2
    ampo .+= -t, "Cdagdn",2,"Cdn",1
    
    if U ≠ 0
        ampo .+= 1/4*U, "Nupdn", 1
        ampo .+= 1/4*U, "Nupdn", 2
    end
    if μ ≠ 0
        ampo .+= -1/4*μ, "Nup", 1
        ampo .+= -1/4*μ, "Ndn", 1
        ampo .+= -1/4*μ, "Nup", 2
        ampo .+= -1/4*μ, "Ndn", 2
    end
    H = MPO(ampo,sites)

    H1 = Array(H[1],inds(H[1])...)
    H2 = Array(H[2],inds(H[2])...)
    h = reshape(ein"aij,apq->ipjq"(H1,H2),16,16)

    return h
end

@doc raw"
    THubbard(t::Real,U::Real,μ::Real)

return a struct representing THubbard model
"
struct THubbard{T<:Real} <: HamiltonianModel
    t::T
	U::T
    μ::T
end

"""
	hamiltonian(model::Hubbard)
"""
function hamiltonian(model::THubbard)
	t = model.t
	U = model.U
    μ = model.μ
    ampo = AutoMPO()
    sites = siteinds("Electron",2)
    ampo .+= -t, "Cdagup",1,"Cdn",2
    ampo .+= -t, "Cdagdn",2,"Cup",1
    ampo .+= -t, "Cdagdn",1,"Cup",2
    ampo .+= -t, "Cdagup",2,"Cdn",1
    
    if U ≠ 0
        ampo .+= 1/4*U, "Nupdn", 1
        ampo .+= 1/4*U, "Nupdn", 2
    end
    if μ ≠ 0
        ampo .+= -1/4*μ, "Nup", 1
        ampo .+= -1/4*μ, "Ndn", 1
        ampo .+= -1/4*μ, "Nup", 2
        ampo .+= -1/4*μ, "Ndn", 2
    end
    H = MPO(ampo,sites)

    H1 = Array(H[1],inds(H[1])...)
    H2 = Array(H[2],inds(H[2])...)
    h = reshape(ein"aij,apq->ipjq"(H1,H2),16,16)

    return h
end

@doc raw"
    hop_pair(t::Real,γ::Real)

return a struct representing hop_pair model
"
struct hop_pair{T<:Real} <: HamiltonianModel
    t::T
	γ::T
end

"""
	hamiltonian(model::hop_pair)
"""
function hamiltonian(model::hop_pair)
	t = model.t
    γ = model.γ
    ampo = AutoMPO()
    sites = siteinds("Electron",2)
    ampo .+= -t, "Cdagup",1,"Cup",2
    ampo .+= -t, "Cdagup",2,"Cup",1
    ampo .+= -t, "Cdagdn",1,"Cdn",2
    ampo .+= -t, "Cdagdn",2,"Cdn",1
    
    ampo .+= γ, "Cdagup",1,"Cdagdn",2
    ampo .+= -γ, "Cdagdn",1,"Cdagup",2
    ampo .+= γ, "Cdn",2,"Cup",1
    ampo .+= -γ, "Cup",2,"Cdn",1
    H = MPO(ampo,sites)

    H1 = Array(H[1],inds(H[1])...)
    H2 = Array(H[2],inds(H[2])...)
    h = reshape(ein"aij,apq->ipjq"(H1,H2),16,16)

    return h
end

function Hubbard_hand(model::Hubbard) where T <: Real
    t = model.t
	U = model.U
    μ = model.μ
    H = zeros(4,4,4,4)
    H .+= -t * (ein"ab,cd -> acbd"(Cup',Cup) .+ ein"ab,cd -> acbd"(Cup,Cup'))
    H .+= -t * (ein"ab,cd -> acbd"(Cdn',Cdn) .+ ein"ab,cd -> acbd"(Cdn,Cdn'))
    H .+= U/4 * (ein"ab,cd -> acbd"(Nup*Ndn,I(4)) .+ ein"ab,cd -> acbd"(I(4),Nup*Ndn))
    H .+= -μ/4 * (ein"ab,cd -> acbd"(Nup.+Ndn,I(4)) .+ ein"ab,cd -> acbd"(I(4),Nup.+Ndn))
    return reshape(H,16,16)
end

function hop_pair_hand(t::T,γ::T) where T <: Real
    H = zeros(4,4,4,4)
    H .+= -t * (ein"ab,cd -> acbd"(Cup',Cup) .+ ein"ab,cd -> acbd"(Cup,Cup'))
    H .+= -t * (ein"ab,cd -> acbd"(Cdn',Cdn) .+ ein"ab,cd -> acbd"(Cdn,Cdn'))
    H .+= γ * (ein"ab,cd -> acbd"(Cup',Cdn') .- ein"ab,cd -> acbd"(Cdn',Cup'))
    H .+= γ * (ein"ab,cd -> acbd"(Cup,Cdn) .- ein"ab,cd -> acbd"(Cdn,Cup))
    return reshape(H,16,16)
end

