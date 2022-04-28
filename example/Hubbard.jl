using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using VUMPS

CUDA.allowscalar(false)
Random.seed!(100)
model = Hubbard(1.0, 12.0, 6.0)
symmetry = :U1
folder = "./example/Hubbard/$symmetry/particle/"
ipeps, key = init_ipeps(model; Ni=2, Nj=2, symmetry=symmetry, atype=Array, folder=folder, D=4, χ=20, tol=1e-10, maxiter=10)
# folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter = key
# key = folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter
# consts = initial_consts(key)
# double_ipeps_energy(atype(ipeps), consts, key)
optimiseipeps(ipeps, key; f_tol = 1e-10, opiter = 10, verbose = true)