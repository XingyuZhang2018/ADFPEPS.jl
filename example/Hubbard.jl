using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using VUMPS

CUDA.allowscalar(false)
Random.seed!(100)
model = Hubbard(1.0, 12.0, 6.0)
symmetry = :U1
atype = Array
folder = "./example/Hubbard/$symmetry/particle/"
indD = [0, 1, 2]
indχ = [-2, -1, 0, 1, 2]
dimsD = [1, 2, 1]
dimsχ = [1, 4, 6, 4, 1]
D = sum(dimsD)
χ = sum(dimsχ)
ipeps, key = init_ipeps(model; Ni=2, Nj=2, symmetry=symmetry, atype=atype, folder=folder, tol=1e-10, maxiter=10,
D=D, χ=χ, indD = indD, indχ = indχ, dimsD = dimsD, dimsχ = dimsχ)
# consts = initial_consts(key)
# double_ipeps_energy(atype(ipeps), consts, key)
optimiseipeps(ipeps, key; f_tol = 1e-10, opiter = 100, verbose = true)