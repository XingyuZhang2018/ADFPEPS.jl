using ADFPEPS
using ADFPEPS: initλΓ, evoGate, update_row!, update_column!
using TeneT
using Test

@testset "SU init" for symmetry in [:none, :U1], stype in [tJZ2()], atype in [Array], dtype in [ComplexF64]
    D = 3
    d = 2
    ST = SymmetricType(Val(symmetry), stype, atype, dtype)
    λ, Γ = initλΓ(ST, D, d)
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
    λ, Γ = initλΓ(ST, D, d)
    λo, Γo = copy(λ), copy(Γ)
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
    λ, Γ = initλΓ(ST, D, d)
    λo, Γo = copy(λ), copy(Γ)
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
    λ, Γ = initλΓ(ST, D, d)
    λo, Γo = copy(λ), copy(Γ)
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
    λ, Γ = initλΓ(ST, D, d)
    λo, Γo = copy(λ), copy(Γ)
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