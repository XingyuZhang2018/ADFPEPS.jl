using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules, U1swapgate
using CUDA
using Random
using TeneT
using TeneT: randU1, zerosU1, IU1, qrpos, lqpos,svd!, initialA, zerosinitial, asU1Array, zerosinitial, randU1DiagMatrix, invDiagU1Matrix, sqrtDiagU1Matrix
using LinearAlgebra
using OMEinsum
using Printf 
using NPZ

atype = Array
dtype = ComplexF64
d = 9

sitetype = tJbilayerZ2()
model = tJ_bilayer(3.0,1.0,0.0,2.0,-2.0)
ipeps = atype{dtype}(npzread("./example/U1ipeps_D2chi8_t3.000_J1.000_Jab2.000_mu-2.000.npy") )   

symmetry = :U1
tol = 1e-10
maxiter = 50
miniter = 1
indD, dimsD = [0,1], [2,2]
indχ, dimsχ = [0,1], [8,8]

χ = sum(dimsχ)
D = sum(dimsD)
folder = "./data/$sitetype/$(model)_$(D)/"
key = (folder, model, 2, 2, symmetry, sitetype, atype, d, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ)
consts = initial_consts(key)

E = double_ipeps_energy(ipeps, consts, key)
println("E = \n", E)