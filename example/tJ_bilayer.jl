using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using TeneT

CUDA.allowscalar(false)
Random.seed!(100)

 indD = [0,1]
dimsD = [2,2]
 indχ = [0,1]
dimsχ = [5,5]

sitetype = tJZ2()

ipeps,key = init_ipeps(tJ_bilayer(3.0,1.0,0.0,2.0,0.0); 
                       Ni = 1, 
                       Nj = 1, 
                 sitetype = sitetype,
                    atype = Array, 
                   folder = "./example/$siteinds/",
                      tol = 1e-10, 
                  maxiter = 10, 
                  miniter = 1, 
                        d = 9,      
                        D = sum(dimsD), 
                        χ = sum(dimsχ), 
                     indD = indD, 
                     indχ = indχ, 
                    dimsD = dimsD, 
                    dimsχ = dimsχ)
# consts = initial_consts(key);
# double_ipeps_energy(ipeps, consts, key);
optimiseipeps(ipeps, key; 
                f_tol = 1e-10, 
               opiter = 100, 
           maxiter_ad = 10,
           miniter_ad = 3,
              verbose = true)