using ADFPEPS
using ADFPEPS:bulk,HORIZONTAL_RULES
using Random
using Test
using OMEinsum

@testset "ADFPEPS:bulk" begin
    Random.seed!(100)
    T = rand(2,2,4,2,2)
    # @show bulk(T)
    @show 
end
