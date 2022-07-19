using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using VUMPS
# using ProfileView

CUDA.allowscalar(false)
Random.seed!(100)
model = hop_pair(1.0, 1.0)
symmetry = :U1
atype = CuArray
folder = "./example/$model/$symmetry/"
indD = [0, 1, 2, 3, 4, 5]
indχ = collect(-2:2)
dimsD = [1, 1, 2, 2, 1, 1]
dimsχ = [10, 20, 40, 20, 10]
D = sum(dimsD)
χ = sum(dimsχ)
ipeps, key = init_ipeps(model; Ni=1, Nj=1, symmetry=symmetry, atype=atype, folder=folder, tol=1e-10, maxiter=10, D=D, χ=χ, indD = indD, indχ = indχ, dimsD = dimsD, dimsχ = dimsχ)
# consts = initial_consts(key)
# double_ipeps_energy(atype(ipeps), consts, key)
# ProfileView.@profview optimiseipeps(ipeps, key; f_tol = 1e-10, opiter = 0, verbose = true)
optimiseipeps(ipeps, key; f_tol = 1e-10, opiter = 100, verbose = true)