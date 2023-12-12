using ADFPEPS
using ADFPEPS: evoGate, update_row!, update_column!, update_once_2nd!, back_to_state, initial_consts, double_ipeps_energy
using TeneT
using Test
using Random

@testset "SU init" for symmetry in [:none, :U1], stype in [tJZ2()], atype in [Array], dtype in [ComplexF64]
    D = 3
    d = 2
    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    Γ, λ = initΓλ(ST, D, d)
    @test size(λ) == (4,)
    @test size(Γ) == (2,)
    @test size(λ[1]) == (D,D)
    @test size(Γ[1]) == (d,D,D,D,D)
end

@testset "SU evoGate" for symmetry in [:U1], stype in [tJZ2()], atype in [Array], dtype in [ComplexF64]
    model = tJ(1.0,1.0,1.0)
    d = 3

    dτ = 0.1
    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    U_2sites, U_local = evoGate(ST, model, dτ)
    
    @test size(U_2sites) == (d,d,d,d)
    @test size(U_local) == (d,d) 
end

@testset "SU update_row!" for symmetry in [:U1], stype in [tJZ2()], atype in [Array], dtype in [ComplexF64]
    model = tJ(1.0,1.0,1.0)
    d = 3
    D = 3
    dτ = 0.1
    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    U_2sites, U_local = evoGate(ST, model, dτ)

    # update right bond, or 1
    Γ, λ = initΓλ(ST, D, d)
    Γo, λo = copy(Γ), copy(λ)
    update_row!(ST, Γ, λ, U_2sites, D; whichbond = "right")
    @test size(λ[1]) == (D,D)
    @test size(Γ[1]) == (d,D,D,D,D)
    @test size(Γ[2]) == (d,D,D,D,D)
    
    @test λ[1] != λo[1]
    @test λ[2] == λo[2]
    @test λ[3] == λo[3]
    @test λ[4] == λo[4]

    @test Γ[1] != Γo[1]
    @test Γ[2] != Γo[2]

    # update left bond, or 3
    Γ, λ = initΓλ(ST, D, d)
    Γo, λo = copy(Γ), copy(λ)
    update_row!(ST, Γ, λ, U_2sites, D; whichbond = "left")
    @test size(λ[3]) == (D,D)
    @test size(Γ[1]) == (d,D,D,D,D)
    @test size(Γ[2]) == (d,D,D,D,D)
    
    @test λ[1] == λo[1]
    @test λ[2] == λo[2]
    @test λ[3] != λo[3]
    @test λ[4] == λo[4]

    @test Γ[1] != Γo[1]
    @test Γ[2] != Γo[2]
end

@testset "SU update_column!" for symmetry in [:U1], stype in [tJZ2()], atype in [Array], dtype in [ComplexF64]
    model = tJ(1.0,1.0,1.0)
    d = 3
    D = 3
    dτ = 0.1
    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    U_2sites, U_local = evoGate(ST, model, dτ)

    # update up bond, or 4
    Γ, λ = initΓλ(ST, D, d)
    Γo, λo = copy(Γ), copy(λ)
    update_column!(ST, Γ, λ, U_2sites, D; whichbond = "up")
    @test size(λ[4]) == (D,D)
    @test size(Γ[1]) == (d,D,D,D,D)
    @test size(Γ[2]) == (d,D,D,D,D)
    
    @test λ[1] == λo[1]
    @test λ[2] == λo[2]
    @test λ[3] == λo[3]
    @test λ[4] != λo[4]

    @test Γ[1] != Γo[1]
    @test Γ[2] != Γo[2]

    # update down bond, or 2
    Γ, λ = initΓλ(ST, D, d)
    Γo, λo = copy(Γ), copy(λ)
    update_column!(ST, Γ, λ, U_2sites, D; whichbond = "down")
    @test size(λ[2]) == (D,D)
    @test size(Γ[1]) == (d,D,D,D,D)
    @test size(Γ[2]) == (d,D,D,D,D)
    
    @test λ[1] == λo[1]
    @test λ[2] != λo[2]
    @test λ[3] == λo[3]
    @test λ[4] == λo[4]

    @test Γ[1] != Γo[1]
    @test Γ[2] != Γo[2]
end

@testset "SU update_once_2nd!" for symmetry in [:U1], stype in [tJZ2()], atype in [Array], dtype in [ComplexF64]
    model = tJ(1.0,1.0,1.0)
    d = 3
    D = 3
    dτ = 0.1
    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    U_2sites, U_local = evoGate(ST, model, dτ)

    Γ, λ = initΓλ(ST, D, d)
    Γo, λo = copy(Γ), copy(λ)
    update_once_2nd!(ST, Γ, λ, U_local, U_2sites, D, true)
    @test all(size(λ[i]) == (D,D) for i in 1:4)
    @test all(size(Γ[i]) == (d,D,D,D,D) for i in 1:2)
    
    @test all(λ[i] != λo[i] for i in 1:4)
    @test all(Γ[i] != Γo[i] for i in 1:2)
end

@testset "SU back_to_state" for symmetry in [:U1], stype in [tJZ2()], atype in [Array], dtype in [ComplexF64]
    d = 3
    D = 4
    ST = SymmetricType(Val(symmetry), stype, atype, dtype)

    Γ, λ = initΓλ(ST, D, d)
    A, B = back_to_state(Γ, λ)
    @test size(A) == (D,D,d,D,D)
    @test size(B) == (D,D,d,D,D)
end

@testset "SU update_ABBA!" for symmetry in [:U1], stype in [tJZ2()], atype in [Array], dtype in [ComplexF64]
    Random.seed!(42)
    model = tJ(3.0,1.0,-3.0)
    d = 3
    D = 2
    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    algorithm = SU(NoUp = 200)

    Γ, λ = initΓλ(ST, D, d)
    Γo, λo = copy(Γ), copy(λ)
    Energy, Entropy = update_ABBA!(algorithm, ST, Γ, λ, model)
    @test all(size(λ[i]) == (D,D) for i in 1:4)
    @test all(size(Γ[i]) == (d,D,D,D,D) for i in 1:2)
    
    @test all(λ[i] != λo[i] for i in 1:4)
    @test all(Γ[i] != Γo[i] for i in 1:2)

    @test Energy[end] ≈ -4.31491649
    @test Entropy[end] ≈ 1.81830002
    
    A, B = back_to_state(Γ, λ)
    L = length(A.tensor)
    ipeps = zeros(dtype, L, 2)  
    ipeps[:,1] = A.tensor 
    ipeps[:,2] = B.tensor  

    indD = [0,1]
    dimsD = [1,1]
    indχ = [0,1]
    dimsχ = [10,10]
    χ = sum(dimsχ)
    Ni, Nj = 2, 2
    tol = 1e-8
    maxiter = 50
    miniter = 1
    folder = "./data/$stype/"
    folder = folder*"/$(model)_$(Ni)x$(Nj)_$(indD)_$(dimsD)_$(indχ)_$(dimsχ)/"
    key = (folder, model, Ni, Nj, symmetry, stype, atype, d, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ)
    consts = initial_consts(key)
    E = double_ipeps_energy(ipeps, consts, key)	
    @test E ≈ -4.06699440
end