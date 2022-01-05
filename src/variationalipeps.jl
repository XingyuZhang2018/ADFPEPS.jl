using FileIO
using HDF5
using Optim, LineSearches
using LinearAlgebra: I, norm
using TimerOutputs
using Zygote

export init_ipeps
export optimiseipeps

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)

Initial `ipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel; Ni::Int, Nj::Int, folder = "./data/", atype = Array, D::Int, χ::Int, tol::Real, maxiter::Int, verbose = true)
    key = (folder, model, Ni, Nj, atype, D, χ, tol, maxiter)
    folder = folder*"/$(model)_$(Ni)x$(Nj)/"
    mkpath(folder)
    chkp_file = folder*"D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).jld2"
    if isfile(chkp_file)
        ipeps = load(chkp_file)["ipeps"]
        verbose && println("load iPEPS from $chkp_file")
    else
        ipeps = rand(ComplexF64,D,D,4,D,D,Int(ceil(Ni*Nj/2)))
        verbose && println("random initial iPEPS $chkp_file")
    end
    ipeps /= norm(ipeps)
    return ipeps, key
end

"""
    optimiseipeps(ipeps, h; χ, tol, maxiter, optimargs = (), optimmethod = LBFGS(m = 20))

return the tensor `bulk'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimiseipeps(ipeps::AbstractArray, key; f_tol = 1e-6, opiter = 100, verbose= false, optimmethod = LBFGS(m = 20)) where LT
    folder, model, Ni, Nj, atype, D, χ, tol, maxiter = key
    to = TimerOutput()
    f(x) = @timeit to "forward" double_ipeps_energy(atype(x), key)
    ff(x) = double_ipeps_energy(atype(x), key)
    g(x) = @timeit to "backward" Zygote.gradient(ff,atype(x))[1]
    res = optimize(f, g, 
        ipeps, optimmethod,inplace = false,
        Optim.Options(f_tol=f_tol, iterations=opiter,
        extended_trace=true,
        callback=os->writelog(os, key)),
        )
    println(to)
    return res
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, key=nothing)
    message = "$(round(os.metadata["time"],digits=2))   $(os.iteration)   $(os.value)   $(os.g_norm)\n"

    printstyled(message; bold=true, color=:red)
    flush(stdout)

    folder, model, Ni, Nj, atype, D, χ, tol, maxiter = key
    !(isdir(folder*"/$(model)_$(Ni)x$(Nj)/")) && mkdir(folder*"/$(model)_$(Ni)x$(Nj)/")
    if !(key === nothing)
        logfile = open(folder*"$(model)_$(Ni)x$(Nj)/D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).log", "a")
        write(logfile, message)
        close(logfile)
        save(folder*"$(model)_$(Ni)x$(Nj)/D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).jld2", "ipeps", os.metadata["x"])
    end
    return false
end