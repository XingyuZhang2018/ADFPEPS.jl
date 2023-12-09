using ADFPEPS
using ADFPEPS: initλΓ, evoGate
using TeneT
using Test

@testset "SU init" for symmetry in [:none, :U1], stype in [tJZ2()], atype in [Array], dtype in [ComplexF64]
    D = 3
    d = 2
    SD = SymmetricType(Val(symmetry), stype, atype, dtype)
    λ, Γ = initλΓ(SD, D, d)
    @test size(λ) == (4,)
    @test size(Γ) == (2,)
    @test size(λ[1]) == (D,D)
    @test size(Γ[1]) == (d,D,D,D,D)
end

@testset "SU evoGate" for symmetry in [:U1], stype in [tJZ2()], atype in [Array], dtype in [ComplexF64]
    model = tJ(1.0,1.0,1.0)
    d = 3
    
    dτ = 0.1
    SD = SymmetricType(Val(symmetry), stype, atype, dtype)
    U_local, U_2sites_1 = evoGate(SD, model, dτ)
    
    @test size(U_local) == (d,d)
    @test size(U_2sites_1) == (d,d,d,d)
end