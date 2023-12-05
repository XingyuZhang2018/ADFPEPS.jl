using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using TeneT

CUDA.allowscalar(false)
Random.seed!(100)

 indD = [0,1]
dimsD = [1,1]
 indχ = [0,1]
dimsχ = [10,10]
sitetype = electronZ2()

ipeps,key = init_ipeps(hop_pair(1.0,0.0); 
                       Ni = 2, 
                       Nj = 2, 
                 sitetype = sitetype,
                    atype = Array, 
                   folder = "../data/$sitetype/",
                      tol = 1e-10, 
                  maxiter = 50, 
                  miniter = 1, 
                        d = 4,      
                        D = sum(dimsD), 
                        χ = sum(dimsχ), 
                     indD = indD, 
                     indχ = indχ, 
                    dimsD = dimsD, 
                    dimsχ = dimsχ)
# @show ipeps
# consts = initial_consts(key);
# double_ipeps_energy(ipeps, consts, key);
optimiseipeps(ipeps, key; 
                f_tol = 1e-10, 
               opiter = 100, 
           maxiter_ad = 10,
           miniter_ad = 1,
              verbose = true)