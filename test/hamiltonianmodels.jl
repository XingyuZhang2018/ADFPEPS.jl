using ADFPEPS
using Test
using ITensors
using OMEinsum
using LinearAlgebra

@testset "hamiltonian" for atype in [Array]
    Random.seed!(100)
    model = ADFPEPS.SpinfulFermions(0.0,4.0)
    @test diag(ADFPEPS.hamiltonian(model)) == 0.5*[1,0,0,1,0,-1,-1,0,0,-1,-1,0,1,0,0,1]
end