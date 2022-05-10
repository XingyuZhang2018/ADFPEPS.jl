using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using VUMPS

CUDA.allowscalar(false)
Random.seed!(100)
model = THubbard(1.0, 12.0, 6.0)
symmetry = :U1
folder = "./example/THubbard/$symmetry/Sz/"
indD = [0, 1]
indχ = [-1, 0, 1]
dimsD = [1, 1]
dimsχ = [3, 4, 3]
D = sum(dimsD)
χ = sum(dimsχ)
ipeps, key = init_ipeps(model; Ni=2, Nj=2, symmetry=symmetry, atype=Array, folder=folder, tol=1e-10, maxiter=10,
D=D, χ=χ, indD = indD, indχ = indχ, dimsD = dimsD, dimsχ = dimsχ)
# consts = initial_consts(key)
# double_ipeps_energy(atype(ipeps), consts, key)
optimiseipeps(ipeps, key; f_tol = 1e-10, opiter = 100, verbose = true)