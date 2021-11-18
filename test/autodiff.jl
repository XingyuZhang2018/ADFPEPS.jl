using ADFPEPS
using ADFPEPS: parity_conserving
using LinearAlgebra
using Test
using VUMPS:num_grad
using Zygote
CUDA.allowscalar(false)

@testset "parity_conserving" for atype in [Array]
    Random.seed!(100)
    Nv = 2

    T = atype(rand(ComplexF64,2^Nv,2^Nv,4,2^Nv,2^Nv))
    foo(T) = norm(parity_conserving(T))
    @test Zygote.gradient(foo, T)[1] ≈ num_grad(foo, T) atol = 1e-8
end