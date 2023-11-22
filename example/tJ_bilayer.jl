using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using TeneT
using LineSearches
using Optim

CUDA.allowscalar(false)
Random.seed!(42)

 indD = [0,1]
dimsD = [1,1]
 indχ = [-2,-1,0,1,2]
dimsχ = [2,3,4,3,2]
sitetype = tJbilayerSz()

ipeps,key = init_ipeps(tJ_bilayer(3.0,1.0,0.0,2.0,0.0); 
                       Ni = 1, 
                       Nj = 1, 
                 sitetype = sitetype,
                    atype = Array, 
                   folder = "./data/$sitetype/",
                      tol = 1e-10, 
                  maxiter = 50, 
                  miniter = 1, 
                        d = 9,      
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
           miniter_ad = 3,
              verbose = true,
              optimmethod = LBFGS(m = 20,
                alphaguess=LineSearches.InitialStatic(alpha=1e-5,scaled=true)))