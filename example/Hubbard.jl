using ADFPEPS
using ADFPEPS:double_ipeps_energy
using CUDA
using Random

CUDA.allowscalar(false)
Random.seed!(100)
model = Hubbard(1.0,12.0,6.0)
folder = "E:/1 - research/4.9 - AutoDiff/data/ADFPEPS/Hubbard/21/"
ipeps, key = init_ipeps(model; Ni = 2, Nj = 2, symmetry = :none, atype = Array, folder = folder, D=3, χ=12, tol=1e-10, maxiter=10)
# folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter = key
# key = folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter
# double_ipeps_energy(atype(ipeps), key)
optimiseipeps(ipeps, key; f_tol = 1e-6, opiter = 100, verbose = true)