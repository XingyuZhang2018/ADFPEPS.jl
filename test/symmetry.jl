using ADFPEPS
using ADFPEPS: swapgate, fdag, bulk, U1swapgate
using TeneT
using Random
using Test
using OMEinsum

@testset "swapgate with $symmetry symmetry" for symmetry in [:U1], sitetype in [tJZ2(), tJbilayerZ2()]
    D = 3
    sg = swapgate(sitetype, Array, ComplexF64, 4, D)
    sgsymmetry = asSymmetryArray(sg, Val(symmetry), sitetype; dir = [-1,-1,1,1])
    sgsymmetryt = asArray(sitetype, sgsymmetry)
    U1sg = U1swapgate(Array, ComplexF64, 4, D; indqn = getqrange(sitetype, 4, D, 4, D), indims = getblockdims(sitetype, 4, D, 4, D), ifZ2=sitetype.ifZ2)
    @test sg == sgsymmetryt
    @test sgsymmetry == U1sg
end

@testset "hamiltonian with $symmetry symmetry" for symmetry in [:U1], sitetype in [tJZ2(), tJbilayerZ2()]
    Random.seed!(100)
    model = tJ_bilayer(3.0,1.0,0.0,2.0,-1.0)
    h = hamiltonian(model)[1]
    hsymmetry = asSymmetryArray(h, Val(symmetry), sitetype; dir = [-1,-1,1,1])
    hsymmetryt = asArray(sitetype, hsymmetry)
    @test h == hsymmetryt
end