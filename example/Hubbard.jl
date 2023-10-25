using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using VUMPS

CUDA.allowscalar(false)
Random.seed!(100)

 indD = [0, 1]
 indχ = [-2, -1, 0, 1, 2]
dimsD = [1, 1]
dimsχ = [1, 4, 6, 4, 1]

ipeps,key = init_ipeps(Hubbard(1.0, 0.0, 0.0); 
                       Ni = 2, 
                       Nj = 2, 
                 symmetry = :none, 
                    atype = Array, 
                   folder = "./example/$symmetry/",
                      tol = 1e-10, 
                  maxiter = 10, 
                  miniter = 1, 
                        D = sum(dimsD), 
                        χ = sum(dimsχ), 
                     indD = indD, 
                     indχ = indχ, 
                    dimsD = dimsD, 
                    dimsχ = dimsχ)
# consts = initial_consts(key)
# double_ipeps_energy(atype(ipeps), consts, key)
optimiseipeps(ipeps, key; 
                f_tol = 1e-10, 
               opiter = 100, 
              verbose = true)