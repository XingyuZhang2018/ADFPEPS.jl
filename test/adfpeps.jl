using ADFPEPS
using CUDA
using Random

CUDA.allowscalar(false)
Random.seed!(100)
model = ADFPEPS.SpinfulFermions(1.0,0.0)
folder = "E:/1 - research/4.9 - AutoDiff/data/ADFPEPS/"
ipeps, key = init_ipeps(model;atype = Array, folder = folder, D=2, χ=20, tol=1e-10, maxiter=10)
res = optimiseipeps(ipeps, key; f_tol = 1e-6, opiter = 0, verbose = true)