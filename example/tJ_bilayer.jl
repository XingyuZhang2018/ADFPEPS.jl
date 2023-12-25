using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using TeneT
using LineSearches
using Optim

CUDA.allowscalar(false)
Random.seed!(100)

  qnD = [0,1]
dimsD = [2,2]
  qnχ = [0,1]
dimsχ = [8,8]
sitetype = tJbilayerZ2()

ipeps,key = init_ipeps(tJ_bilayer(3.0,1.0,0.0,2.0,-2.0); 
                       Ni = 2, 
                       Nj = 2, 
                   SUinit = false,
                     NoUp = 1000,
                       dτ = 0.4,
                 sitetype = sitetype,
                    atype = Array, 
                   folder = "../data/$sitetype/",
                      tol = 1e-10, 
                  maxiter = 50, 
                  miniter = 1, 
                        d = 9,      
                        D = sum(dimsD), 
                        χ = sum(dimsχ), 
                      qnD =  qnD,  
                      qnχ =  qnχ, 
                    dimsD = dimsD, 
                    dimsχ = dimsχ)
# @show ipeps
ipeps = Array{ComplexF64}(npzread("./example/U1ipeps_D2chi8_t3.000_J1.000_Jab2.000_mu-2.000.npy") )
# consts = initial_consts(key);
# double_ipeps_energy(ipeps, consts, key);
# observable(ipeps, key)
optimiseipeps(ipeps, key; 
                f_tol = 1e-10, 
               opiter = 100, 
           maxiter_ad = 10,
           miniter_ad = 3,
              verbose = true,
          optimmethod = LBFGS(m = 20,
					 alphaguess = LineSearches.InitialStatic(alpha=1e-5,scaled=true)))