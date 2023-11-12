using ADFPEPS
using ADFPEPS: swapgate, Z2swapgate, U1swapgate
using TeneT
using Random
using Test
using OMEinsum

@testset "swapgate" for atype in [Array], dtype in [ComplexF64], siteinds in [tJZ2(),tJSz()] 
    d,D=9,10
    S_none = swapgate(siteinds, atype, dtype, d,D)
    S_U1 = U1swapgate(atype, dtype, d,D; 
                        indqn=getqrange(siteinds, d,D,d,D), 
                        indims=getblockdims(siteinds, d,D,d,D),
                        siteinds.ifZ2
                        )
    @test S_none == asArray(siteinds, S_U1)
    @test asU1Array(siteinds, S_none; dir=[-1,-1,1,1]) == S_U1
end