using ADFPEPS
using ADFPEPS:double_ipeps_energy
using CUDA
using Random

CUDA.allowscalar(false)
Random.seed!(100)
model = ADFPEPS.SpinfulFermions(0.0,-4.0)
folder = "E:/1 - research/4.9 - AutoDiff/data/ADFPEPS/"
ipeps, key = init_ipeps(model; Ni = 2, Nj = 2, atype = Array, folder = folder, D=2, χ=20, tol=1e-10, maxiter=10)
folder, model, atype, D, χ, tol, maxiter = key
res = optimiseipeps(ipeps, key; f_tol = 1e-6, opiter = 100, verbose = true)