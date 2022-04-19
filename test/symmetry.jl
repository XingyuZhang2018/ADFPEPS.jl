using ADFPEPS
using ADFPEPS: swapgate, fdag, bulk
using VUMPS
using Random
using Test
using OMEinsum

@testset "swapgate with $symmetry symmetry" for symmetry in [:Z2, :U1]
    sg = swapgate(4, 4)
    sgsymmetry = asSymmetryArray(sg, Val(symmetry); dir = [-1,-1,1,1])
    sgsymmetryt = asArray(sgsymmetry)
    @test sg == sgsymmetryt

    h = [0.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2]
    hs = asSymmetryArray(h, Val(symmetry); dir = [-1,1])
    hst = asArray(hs)
    @test h == hst
    @show sgsymmetry
end

@testset "hamiltonian with $symmetry symmetry" for symmetry in [:Z2, :U1]
    Random.seed!(100)
    model = Hubbard(1.0,0.0,0.0)
    h = reshape(hamiltonian(model), 4, 4, 4, 4)
    hsymmetry = asSymmetryArray(h, Val(symmetry); dir = [-1,-1,1,1])
    hsymmetryt = asArray(hsymmetry)
    @test h == hsymmetryt
end