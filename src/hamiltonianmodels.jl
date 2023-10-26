using ITensors
using OMEinsum

abstract type HamiltonianModel end

@doc raw"
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"
function hamiltonian end
export hamiltonian_hand

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
    ampo .+= t, "Cdagdn",1,"Cup",2
    ampo .+= t, "Cdagup",2,"Cdn",1
    
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

function hamiltonian_hand(model::Hubbard)
    t = model.t
	U = model.U
    μ = model.μ
    H = zeros(4,4,4,4)

    # t
    H[1,2,2,1], H[1,4,2,3], H[2,1,1,2], H[2,3,1,4] = -t, -t, -t, -t
    H[1,3,3,1], H[2,3,4,1], H[3,1,1,3], H[4,1,2,3] = -t, -t, -t, -t
    H[3,2,4,1], H[3,4,4,3], H[4,1,3,2], H[4,3,3,4] = t, t, t, t
    H[1,4,3,2], H[2,4,4,2], H[3,2,1,4], H[4,2,2,4] = t, t, t, t

    # U
    H[1,4,1,4], H[4,1,4,1] = U/4, U/4
    H[2,4,2,4], H[4,2,4,2] = U/4, U/4
    H[3,4,3,4], H[4,3,4,3] = U/4, U/4
    H[4,4,4,4] = U/2

    # μ
    H[1,2,1,2], H[2,1,2,1], H[1,3,1,3], H[3,1,3,1] = -μ/4, -μ/4, -μ/4, -μ/4
    H[2,3,2,3], H[3,2,3,2], H[2,2,2,2], H[3,3,3,3] = -μ/2, -μ/2, -μ/2, -μ/2
    H[1,4,1,4] += -μ/2
    H[4,1,4,1] += -μ/2
    H[2,4,2,4] += -3*μ/4
    H[4,2,4,2] += -3*μ/4
    H[3,4,3,4] += -3*μ/4
    H[4,3,4,3] += -3*μ/4
    H[4,4,4,4] += -μ
    return reshape(H,16,16)
end

function hamiltonian_hand(model::hop_pair)
    t = model.t
	γ = model.γ
    H = zeros(4,4,4,4)

    # t
    H[1,2,2,1], H[1,4,2,3], H[2,1,1,2], H[2,3,1,4] = -t, -t, -t, -t
    H[1,3,3,1], H[2,3,4,1], H[3,1,1,3], H[4,1,2,3] = -t, -t, -t, -t
    H[3,2,4,1], H[3,4,4,3], H[4,1,3,2], H[4,3,3,4] = t, t, t, t
    H[1,4,3,2], H[2,4,4,2], H[3,2,1,4], H[4,2,2,4] = t, t, t, t

    # γ
    H[1,1,2,3], H[2,3,1,1] =  γ, γ
    H[3,1,4,3], H[4,3,3,1] = -γ,-γ
    H[1,2,2,4], H[2,4,1,2] = -γ,-γ
    H[3,2,4,4], H[4,4,3,2] =  γ, γ

    # -γ
    H[1,1,3,2], H[3,2,1,1] = -γ,-γ
    H[2,1,4,2], H[4,2,2,1] = -γ,-γ
    H[1,3,3,4], H[3,4,1,3] = -γ,-γ
    H[2,3,4,4], H[4,4,2,3] = -γ,-γ
    return reshape(H,16,16)
end

@doc raw"
    tJ(t::Real,γ::Real)

return a struct representing tJ model
"
struct tJ{T<:Real} <: HamiltonianModel
    t::T
	J::T
end

"""
	hamiltonian(model::tJ)
"""
function hamiltonian(model::tJ)
	t = model.t
    J = model.J
    ampo = AutoMPO()
    sites = siteinds("tJ",2)
    ampo .+= -t, "Cdagup",1,"Cup",2
    ampo .+= -t, "Cdagup",2,"Cup",1
    ampo .+= -t, "Cdagdn",1,"Cdn",2
    ampo .+= -t, "Cdagdn",2,"Cdn",1
    
    ampo .+= J/2, "S+",1,"S-",2
    ampo .+= J/2, "S-",1,"S+",2
    ampo .+= J, "Sz",1,"Sz",2

    ampo .+= -J/4, "Ntot",1,"Ntot",2

    H = MPO(ampo,sites)

    H1 = Array(H[1],inds(H[1])...)
    H2 = Array(H[2],inds(H[2])...)
    h = reshape(ein"aij,apq->ipjq"(H1,H2),9,9)

    return h
end

function hamiltonian_hand(model::tJ)
    t = model.t
    J = model.J
    H = zeros(9,9)

    H[2,4], H[3,7], H[4,2], H[7,3] = -t, -t, -t, -t
    H[5,5], H[9,9] = J/4, J/4
    H[6,6], H[8,8] = -J/4, -J/4
    H[6,8], H[8,6] = J/2, J/2

    return H
end

function hamiltonian_hand(model::tJ)
    t = model.t
    J = model.J
    H = zeros(9,9)

    H[2,4], H[3,7], H[4,2], H[7,3] = -t, -t, -t, -t
    H[6,8], H[8,6] = J/2, J/2
    H[6,6], H[8,8] = -J/2, -J/2

    return H
end