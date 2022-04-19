using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate
using CUDA
using Random
using VUMPS

CUDA.allowscalar(false)
Random.seed!(100)
model = Hubbard(1.0,4.0,2.0)
symmetry = :none
folder = "./example/Hubbard/$symmetry/"
ipeps, key = init_ipeps(model; Ni=2, Nj=2, symmetry=symmetry, atype=Array, folder=folder, D=2, χ=20, tol=1e-10, maxiter=10)
# folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter = key
# key = folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter

# SdD = atype(swapgate(4, D))
# SDD = atype(swapgate(D, D))
# hx = reshape(atype{ComplexF64}(hamiltonian(model)), 4,4,4,4)
# hy = reshape(atype{ComplexF64}(hamiltonian(model)), 4,4,4,4)

# SdD = asSymmetryArray(SdD, Val(symmetry); dir=[-1,-1,1,1])
# SDD = asSymmetryArray(SDD, Val(symmetry); dir=[-1,-1,1,1])
# hx = asSymmetryArray(hx, Val(symmetry); dir=[-1,-1,1,1])
# hy = asSymmetryArray(hy, Val(symmetry); dir=[-1,-1,1,1])

# consts = (SdD, SDD, hx, hy)
# double_ipeps_energy(atype(ipeps), consts, key)
optimiseipeps(ipeps, key; f_tol = 1e-6, opiter = 100, verbose = true)