using ADFPEPS
using ADFPEPS: parity_conserving
using CUDA
using LinearAlgebra
using Random
using Test
using VUMPS:num_grad
using ADFPEPS:HamiltonianModel
using Zygote
CUDA.allowscalar(false)

@testset "parity_conserving" for atype in [Array], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D = 2
    T = atype(rand(dtype,D,D,D))
    parity_conserving(T)
end