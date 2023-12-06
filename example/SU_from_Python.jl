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
dtype =  ComplexF64
D = 2
d = 9
χ = 20
sitetype = tJbilayerZ2()
model = tJ_bilayer(3.0,1.0,0.0,2.0,0.0)
 
#ipeps = atype{dtype}(npzread("./example/U1ipeps.npy") )  
place = "D:/2020/TN/PEPS/fPEPS-TRY/a_tJ/python_KH_tJ_20231119/run_tJB/data/tJ_bilayer/SU/2site/D=1/t3J1_Jab2_mu" 
#place = "./example"
ipeps = atype{dtype}(npzread(place*"/U1ipeps_D1chi8_t3.000_J1.000_Jab2.000_mu0.000.npy") )   
  
symmetry = :U1
tol = 1e-10
maxiter = 50
miniter = 1 
indD, dimsD = [0,1], [1,1]
indχ = [0,1]
dimsχ = [10, 10]
folder = "../data/$sitetype/$(model)_$(D)/"
key = (folder, model, 2, 2, symmetry, sitetype, atype, d, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ)
consts = initial_consts(key) 
  
 
E = double_ipeps_energy(ipeps, consts, key)	
println("E = \n", E)

 
 