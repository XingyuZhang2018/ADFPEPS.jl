using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using TeneT

CUDA.allowscalar(false)
Random.seed!(100)

 indD = [0, 1]
dimsD = [1, 1]
 indχ = [0, 1]
dimsχ = [18, 22]
sitetype = tJZ2()

ipeps,key = init_ipeps(tJ(3.0,1.0,0.0); 
                       Ni = 1, 
                       Nj = 1, 
                    atype = Array, 
                   folder = "../data/$sitetype/",
                      tol = 1e-8, 
                  maxiter = 50, 
                  miniter = 1, 
                        d = 3,
                        D = sum(dimsD), 
                        χ = sum(dimsχ), 
                     indD = indD, 
                     indχ = indχ, 
                    dimsD = dimsD, 
                    dimsχ = dimsχ)
# folder, model, Ni, Nj, symmetry, sitetype, atype, d, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ = key
# key = folder, model, Ni, Nj, symmetry, sitetype, atype, d, D, sum(dimsχ), tol, maxiter, miniter, indD, indχ, dimsD, [10,10]
# consts = initial_consts(key)
# double_ipeps_energy(atype(ipeps), consts, key)
optimiseipeps(ipeps, key; 
                f_tol = 1e-10, 
               opiter = 00, 
           maxiter_ad = 10, 
           miniter_ad = 3,
              verbose = true)