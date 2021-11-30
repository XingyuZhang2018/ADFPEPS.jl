using ADFPEPS
using ADFPEPS:double_ipeps_energy
using CUDA
using Random

CUDA.allowscalar(false)
Random.seed!(100)
model = Hubbard(1.0,8.0,4.0)
folder = "E:/1 - research/4.9 - AutoDiff/data/ADFPEPS/"
ipeps, key = init_ipeps(model; Ni = 2, Nj = 2, atype = Array, folder = folder, D=3, χ=30, tol=1e-10, maxiter=10)
# folder, model, Ni, Nj, atype, D, χ, tol, maxiter = key
# double_ipeps_energy(ipeps, model; Ni=Ni,Nj=Nj,χ=χ,maxiter=10,infolder=folder,outfolder=folder)
optimiseipeps(ipeps, key; f_tol = 1e-6, opiter = 100, verbose = true)