using ADFPEPS
using ADFPEPS: swapgate, fdag, bulk, U1swapgate
using TeneT
using Random
using Test
using OMEinsum

@testset "swapgate with $symmetry symmetry" for symmetry in [:U1]
    D = 3
    sg = swapgate(4, D)
    sgsymmetry = asSymmetryArray(sg, Val(symmetry); dir = [-1,-1,1,1], indqn = getqrange(4, D, 4, D), indims = getblockdims(4, D, 4, D))
    sgsymmetryt = asArray(sgsymmetry; indqn = getqrange(4, D, 4, D), indims = getblockdims(4, D, 4, D))
    U1sg = U1swapgate(Array, ComplexF64, 4, D; indqn = getqrange(4, D, 4, D), indims = getblockdims(4, D, 4, D))
    @test sg == sgsymmetryt
    @test sgsymmetry == U1sg

    @show U1sg
end

@testset "hamiltonian with $symmetry symmetry" for symmetry in [:U1]
    Random.seed!(100)
    model = Hubbard(1.0,12.0,6.0)
    h = reshape(hamiltonian(model), 4, 4, 4, 4)
    hsymmetry = asSymmetryArray(h, Val(symmetry); dir = [-1,-1,1,1], indqn = getqrange(4, 4, 4, 4), indims = getblockdims(4, 4, 4, 4))
    hsymmetryt = asArray(hsymmetry; indqn = getqrange(4, 4, 4, 4), indims = getblockdims(4, 4, 4, 4))
    @test h == hsymmetryt
    @show hsymmetry
end