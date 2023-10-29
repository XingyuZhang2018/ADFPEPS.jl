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
        ampo .+= -1/4*μ, "Ntot", 1
        ampo .+= -1/4*μ, "Ntot", 2
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
    μ::T
end

"""
	hamiltonian(model::tJ)
"""
function hamiltonian(model::tJ)
	t = model.t
    J = model.J
    μ = model.μ
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

    if μ ≠ 0
        ampo .+= μ, "Ntot", 1
        ampo .+= μ, "Ntot", 2
    end

    H = MPO(ampo,sites)

    H1 = Array(H[1],inds(H[1])...)
    H2 = Array(H[2],inds(H[2])...)
    h = reshape(ein"aij,apq->ipjq"(H1,H2),9,9)

    return h
end

function hamiltonian_hand(model::tJ)
    t = model.t
    J = model.J
    μ = model.μ
    H = zeros(9,9)

    # -t
    for (i,j) in [[2, 4],[3, 7],[4, 2],[7, 3]]
        H[i,j] += -t
    end

    # J
    for (i,j) in [[5,5],[9,9]]
        H[i,j] += J/4
    end
    for (i,j) in [[6,6],[8,8]]
        H[i,j] += -J/4
    end
    for (i,j) in [[6,8],[8,6]]
        H[i,j] += J/2
    end

    # -J/4
    for (i,j) in [[5,5],[6,6],[8,8],[9,9]]
        H[i,j] += -J/4
    end

    # μ
    for (i,j) in [[2,2],[3,3],[4,4],[7,7]]
        H[i,j] += μ
    end
    for (i,j) in [[5,5],[6,6],[8,8],[9,9]]
        H[i,j] += 2*μ
    end

    return H
end

@doc raw"
    tJ_bilayer(t::Real,γ::Real)

return a struct representing tJ_bilayer model
"
struct tJ_bilayer <: HamiltonianModel
    t二::Real
	J二::Real
    t⊥::Real
    J⊥::Real
    μ::Real
end

function hamiltonian(model::tJ_bilayer)
    t二 = model.t二
    J二 = model.J二
    t⊥ = model.t⊥
    J⊥ = model.J⊥
    μ = model.μ

    ampo = AutoMPO()
    sites = siteinds("tJ",4)
    ampo .+= -t二, "Cdagup",1,"Cup",3
    ampo .+= -t二, "Cdagup",3,"Cup",1
    ampo .+= -t二, "Cdagdn",1,"Cdn",3
    ampo .+= -t二, "Cdagdn",3,"Cdn",1
    
    ampo .+= -t二, "Cdagup",2,"Cup",4
    ampo .+= -t二, "Cdagup",4,"Cup",2
    ampo .+= -t二, "Cdagdn",2,"Cdn",4
    ampo .+= -t二, "Cdagdn",4,"Cdn",2

    ampo .+= J二/2, "S+",1,"S-",3
    ampo .+= J二/2, "S-",1,"S+",3
    ampo .+= J二, "Sz",1,"Sz",3

    ampo .+= J二/2, "S+",2,"S-",4
    ampo .+= J二/2, "S-",2,"S+",4
    ampo .+= J二, "Sz",2,"Sz",4
    
    ampo .+= -J二/4, "Ntot",1,"Ntot",3
    ampo .+= -J二/4, "Ntot",2,"Ntot",4

    if μ ≠ 0
        ampo .+= μ, "Ntot",1
        ampo .+= μ, "Ntot",2
        ampo .+= μ, "Ntot",3
        ampo .+= μ, "Ntot",4
    end

    H = MPO(ampo,sites)

    H1 = Array(H[1],inds(H[1])...)
    H2 = Array(H[2],inds(H[2])...)
    H3 = Array(H[3],inds(H[3])...)
    H4 = Array(H[4],inds(H[4])...)
    H二 = reshape(ein"iae,ijbf,jkcg,kdh->cdabghef"(H1,H2,H3,H4),9,9,9,9)

    ampo = AutoMPO()
    sites = siteinds("tJ",2)
    ampo .+= -t⊥, "Cdagup",1,"Cup",2
    ampo .+= -t⊥, "Cdagup",2,"Cup",1
    ampo .+= -t⊥, "Cdagdn",1,"Cdn",2
    ampo .+= -t⊥, "Cdagdn",2,"Cdn",1
    
    ampo .+= J⊥/2, "S+",1,"S-",2
    ampo .+= J⊥/2, "S-",1,"S+",2
    ampo .+= J⊥, "Sz",1,"Sz",2

    H = MPO(ampo,sites)

    H1 = Array(H[1],inds(H[1])...)
    H2 = Array(H[2],inds(H[2])...)
    H⊥ = reshape(ein"aij,apq->ipjq"(H1,H2),9,9)

    return H二,  H⊥
end

kronplus(A) = mapreduce(x->kron(x[1], x[2]), +, A)

function hamiltonian_hand(model::tJ_bilayer)
    t二 = model.t二
    J二 = model.J二
    t⊥ = model.t⊥
    J⊥ = model.J⊥
    μ = model.μ

    Cu = [0 1 0; 0 0 0; 0 0 0]
    Cd = [0 0 1; 0 0 0; 0 0 0]
    Cu1 = kron(Cu, I(3))
    Cu2 = kron(I(3), Cu)
    Cd1 = kron(Cd, I(3))
    Cd2 = kron(I(3), Cd)
    Nu1 = Cu1'*Cu1
    Nu2 = Cu2'*Cu2
    Nd1 = Cd1'*Cd1
    Nd2 = Cd2'*Cd2
    Ntot1 = Nu1 + Nd1
    Ntot2 = Nu2 + Nd2
    Sx = 1/2*(Cu'*Cd + Cd'*Cu)
    Sy = 1im/2*(-Cu'*Cd + Cd'*Cu)
    Sz = 1/2*(Cu'*Cu - Cd'*Cd)
    Sx1 = 1/2*(Cu1'*Cd1 + Cd1'*Cu1)
    Sx2 = 1/2*(Cu2'*Cd2 + Cd2'*Cu2)
    Sy1 = 1im/2*(-Cu1'*Cd1 + Cd1'*Cu1)
    Sy2 = 1im/2*(-Cu2'*Cd2 + Cd2'*Cu2)
    Sz1 = 1/2*(Cu1'*Cu1 - Cd1'*Cd1)
    Sz2 = 1/2*(Cu2'*Cu2 - Cd2'*Cd2)

    Ht = -t二 * mapreduce(x->kron(x[1], x[2]), +,
        [
         [Cu1',Cu1], [Cu1,Cu1'], [Cu2',Cu2], [Cu2,Cu2'],
         [Cd1',Cd1], [Cd1,Cd1'], [Cd2',Cd2], [Cd2,Cd2']
        ]
    )

    #hopping -
    Ht = reshape(Ht,9,9,9,9)
    for (i,j,k,l) in [
        [5,1,2,4],[8,1,2,7],[6,1,3,4],[9,1,3,7],[1,5,2,4],[1,8,2,7],[1,6,3,4],[1,9,3,7],
        [2,4,5,1],[2,7,8,1],[3,4,6,1],[3,7,9,1],[2,4,1,5],[2,7,1,8],[3,4,1,6],[3,7,1,9],

        [5,4,4,5],[8,4,7,5],[6,4,4,6],[9,4,7,6],[2,5,5,2],[2,8,8,2],[2,6,5,3],[2,9,8,3],
        [4,5,5,4],[7,5,8,4],[4,6,6,4],[7,6,9,4],[5,2,2,5],[8,2,2,8],[5,3,2,6],[8,3,2,9],

        [5,7,4,8],[8,7,7,8],[6,7,4,9],[9,7,7,9],[3,5,6,2],[3,8,9,2],[3,6,6,3],[3,9,9,3],
        [4,8,5,7],[7,8,8,7],[4,9,6,7],[7,9,9,7],[6,2,3,5],[9,2,3,8],[6,3,3,6],[9,3,3,9],
        ]
        Ht[i,j,k,l] *= -1
    end
    Ht = reshape(Ht,81,81)

    HJ = J二 * mapreduce(x->kron(x[1], x[2]), +,
        [
         [Sx1,Sx1], [Sy1,Sy1], [Sz1,Sz1],
         [Sx2,Sx2], [Sy2,Sy2], [Sz2,Sz2]
        ]
    )

    HJ4 = -J二/4 * mapreduce(x->kron(x[1], x[2]), +,
        [
         [Ntot1,Ntot1], [Ntot2,Ntot2]
        ]
    )

    Hμ = μ * mapreduce(x->kron(x[1], x[2]), +,
        [
         [Ntot1+Ntot2,I(9)], [I(9),Ntot1+Ntot2]
        ]
    )

    H二 = Ht + HJ + HJ4 + Hμ
    H二 = reshape(H二,9,9,9,9)

    Ht = -t⊥ * mapreduce(x->kron(x[1], x[2]), +,
        [
         [Cu',Cu], [Cu,Cu'],
         [Cd',Cd], [Cd,Cd'],
        ]
    )

    HJ = J⊥ * mapreduce(x->kron(x[1], x[2]), +,
        [
         [Sx,Sx], [Sy,Sy], [Sz,Sz]
        ]
    )

    H⊥ = Ht + HJ

    return H二,  H⊥
end
