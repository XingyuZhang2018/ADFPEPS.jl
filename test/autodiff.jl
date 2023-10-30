using ADFPEPS
using ADFPEPS: parity_conserving, conjipeps, double_ipeps_energy
using CUDA
using LinearAlgebra
using Random
using Test
using TeneT:num_grad
using Zygote
CUDA.allowscalar(false)

@testset "parity_conserving" for atype in [Array], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    Nv = 1
    D = 2^Nv
    T = atype(rand(dtype,D,D,4,D,D,4))
    function foo(T)
        ipeps = reshape([parity_conserving(T[:,:,:,:,:,i]) for i = 1:4], (2, 2))
        norm(ipeps)
    end
    @test Zygote.gradient(foo, T)[1] ≈ num_grad(foo, T) atol = 1e-8
end

@testset "energy" for atype in [Array], dtype in [ComplexF64], Ni = [1], Nj = [1]
    Random.seed!(100)
    folder = "E:/1 - research/4.9 - AutoDiff/data/ADFPEPS/"
    model = Hubbard(1.0,4.0,2.0)
    ipeps, key = init_ipeps(model; Ni = Ni, Nj = Nj, atype = atype, folder = folder, D=2, χ=4, tol=1e-10, maxiter=10)
    function foo(ipeps)
        double_ipeps_energy(ipeps, model; Ni=Ni,Nj=Nj,χ=χ,maxiter=10,infolder=folder,outfolder=folder)
    end
    folder, model, Ni, Nj, atype, D, χ, tol, maxiter = key
    @test Zygote.gradient(foo, ipeps)[1] ≈ num_grad(foo, ipeps) atol = 1e-3
end