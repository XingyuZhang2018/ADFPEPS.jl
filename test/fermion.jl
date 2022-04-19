using ADFPEPS
using ADFPEPS: parity_conserving
using ADFPEPS: swapgate, fdag, bulk
using VUMPS
using Random
using Test
using OMEinsum

@testset "parity_conserving" for atype in [Array], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D = 2
    T = atype(rand(dtype,D,D,D))
    parity_conserving(T)
end

@testset "bulk" begin
    Random.seed!(100)
    D = 2
    A = randinitial(Val(:U1), Array, Float64, D,D,4,D,D; dir = [-1,-1,1,1,1])
    Atensor = asArray(A)
    SDDtensor = swapgate(D, D)
    SDD = asSymmetryArray(SDDtensor, Val(:U1); dir = [-1,-1,1,1])
    @test fdag(A, SDD) !== nothing
    @test asArray(fdag(A, SDD)) ≈ fdag(Atensor, SDDtensor)

    @test bulk(A, SDD) !== nothing
    # @test asArray(bulk(A, SDD)) ≈ bulk(Atensor, SDDtensor) # before reshape
end