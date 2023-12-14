using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using TeneT

CUDA.allowscalar(false)
Random.seed!(100)

  qnD = [0, 1]
dimsD = [1, 1]
  qnχ = [0, 1]
dimsχ = [10, 10]
sitetype = tJZ2()

ipeps,key = init_ipeps(tJ(3.0,1.0,-3.0); 
                       Ni = 2, 
                       Nj = 2, 
                   SUinit = true,
                    atype = Array, 
                   folder = "../data/$sitetype/",
                      tol = 1e-8, 
                  maxiter = 50, 
                  miniter = 1, 
                        d = 3,
                        D = sum(dimsD), 
                        χ = sum(dimsχ), 
                      qnD = qnD, 
                      qnχ = qnχ, 
                    dimsD = dimsD, 
                    dimsχ = dimsχ)
# folder, model, Ni, Nj, symmetry, sitetype, atype, d, D, χ, tol, maxiter, miniter,  qnD,  qnχ, dimsD, dimsχ = key
# key = folder, model, Ni, Nj, symmetry, sitetype, atype, d, D, sum(dimsχ), tol, maxiter, miniter,  qnD,  qnχ, dimsD, [10,10]
# consts = initial_consts(key)
# double_ipeps_energy(atype(ipeps), consts, key)
optimiseipeps(ipeps, key; 
                f_tol = 1e-10, 
               opiter = 100, 
           maxiter_ad = 10, 
           miniter_ad = 3,
              verbose = true)