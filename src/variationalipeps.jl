using FileIO
using HDF5
using Optim, LineSearches
using LinearAlgebra: I, norm
using TimerOutputs
using Zygote

export init_ipeps, initial_consts
export optimiseipeps

"""
	calculate enviroment (E1...E6)
	a ────┬──── c 
	│     b     │ 
	├─ d ─┼─ e ─┤ 
	│     g     │ 
	f ────┴──── h 
	order: adf,abc,dgeb,fgh,ceh
"""
function ipeps_enviroment(M::AbstractArray, key)
	folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter, indD, indχ, dimsD, dimsχ = key

	# VUMPS
	_, ALu, Cu, ARu, ALd, Cd, ARd, FLo, FRo, FL, FR = obs_env(M; χ=χ, maxiter=maxiter, miniter=1, tol = tol, verbose=true, savefile = true, infolder=folder, outfolder=folder, updown = true, downfromup = false, show_every=Inf, U1info = (indD, indχ, dimsD, dimsχ))

	ACu = reshape([ein"abc,cd->abd"(ALu[i],Cu[i]) for i = 1:Ni*Nj], (Ni, Nj))
	ACd = reshape([ein"abc,cd->abd"(ALd[i],Cd[i]) for i = 1:Ni*Nj], (Ni, Nj))
	return FLo, ACu, ARu, FRo, ARd, ACd, FL, FR
end

ABBA(i) = i in [1,4] ? 1 : 2

function buildipeps(ipeps, key)
	folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter, indD, indχ, dimsD, dimsχ = key
	if symmetry == :Z2
		info = Zygote.@ignore zerosinitial(Val(symmetry), atype, ComplexF64, D,D,4,D,D; dir = [-1,-1,1,1,1], q = [1])
		reshape([Z2Array(info.parity, [reshape(atype(ipeps[1 + sum(prod.(info.dims[1:j-1])):sum(prod.(info.dims[1:j])), ABBA(i)]), tuple(info.dims[j]...)) for j in 1:length(info.dims)], info.size, info.dims, 1) for i = 1:Ni*Nj], (Ni, Nj))
	elseif symmetry == :U1
		info = Zygote.@ignore zerosinitial(Val(symmetry), atype, ComplexF64, D,D,4,D,D; 
			dir = [-1,-1,1,1,1], 
			indqn = [indD, indD, getqrange(4)..., indD, indD], 
			indims = [dimsD, dimsD, getblockdims(4)..., dimsD, dimsD], 
			q = [1]
		)
		reshape([U1Array(info.qn, info.dir, atype(ipeps[:, ABBA(i)]), info.size, info.dims, 1) for i = 1:Ni*Nj], (Ni, Nj))
	else
		info = Zygote.@ignore zerosinitial(Val(:Z2), atype, ComplexF64, D,D,4,D,D; dir = [-1,-1,1,1,1], q = [1])
		reshape([asArray(Z2Array(info.parity, [reshape(atype(ipeps[1 + sum(prod.(info.dims[1:j-1])):sum(prod.(info.dims[1:j])), ABBA(i)]), tuple(info.dims[j]...)) for j in 1:length(info.dims)], info.size, info.dims, 1)) for i = 1:Ni*Nj], (Ni, Nj))
	end
end

function double_ipeps_energy(ipeps::AbstractArray, consts, key)	
	folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter, indD, indχ, dimsD, dimsχ = key
    SdD, SDD, hx, hy, HORIZONTAL_RULES, VERTICAL_RULES, reinfo = consts
	T = buildipeps(ipeps, key)
	M = reshape([bulk(T[i], SDD, indD, dimsD) for i = 1:Ni*Nj], (Ni, Nj))
	E1,E2,E3,E4,E5,E6,E7,E8 = ipeps_enviroment(M, key)

	etol = 0
	for j = 1:Nj, i = 1:Ni
		ir = Ni + 1 - i
		jr = j + 1 - (j==Nj) * Nj
		
		Tij, Tijr, Tirj = T[i,j], T[i,jr], T[ir,j]
		ex = (E1[i,j],E2[i,j],E3[i,jr],E4[i,jr],E5[ir,jr],E6[ir,j])
		ρx = square_ipeps_contraction_horizontal(Tij, Tijr, SdD, SDD, ex, HORIZONTAL_RULES, reinfo)
		# ρ1 = reshape(ρ,16,16)
		# @show norm(ρ1-ρ1')
        Ex = ein"ijkl,ijkl -> "(ρx,hx)[]
		nx = dtr(ρx) # nx = ein"ijij -> "(ρx)
		etol += Ex/nx
		println("─ = $(Ex/nx)") 

        ey = (E1[ir,j],E2[i,j],E4[ir,j],E6[i,j],E7[i,j],E8[i,j])
		ρy = square_ipeps_contraction_vertical(Tij, Tirj, SdD, SDD, ey, VERTICAL_RULES, reinfo)
		# ρ1 = reshape(ρ,16,16)
		# @show norm(ρ1-ρ1')
        Ey = ein"ijkl,ijkl -> "(ρy,hy)[]
		ny = dtr(ρy) # ny = ein"ijij -> "(ρy)[]
		etol += Ey/ny
		println("│ = $(Ey/ny)")
	end
	@show etol/Ni/Nj
	return real(etol)/Ni/Nj
end

function square_ipeps_contraction_vertical(T1, T2, SdD, SDD, env, VERTICAL_RULES, reinfo)
	nu,nl,nf,nd,nr = size(T1)
	χ = size(env[1])[1]
	
	E1 = symmetryreshape(env[1], χ,nl,nl,χ; reinfo = reinfo[1])[1]
	E2 = symmetryreshape(env[2], χ,nl,nl,χ; reinfo = reinfo[2])[1]
	E4 = symmetryreshape(env[3], χ,nl,nl,χ; reinfo = reinfo[3])[1]
	E6 = symmetryreshape(env[4], χ,nl,nl,χ; reinfo = reinfo[2])[1]
	E7 = symmetryreshape(env[5], χ,nl,nl,χ; reinfo = reinfo[1])[1]
	E8 = symmetryreshape(env[6], χ,nl,nl,χ; reinfo = reinfo[3])[1]

	result = VERTICAL_RULES(T1,fdag(T1, SDD),SDD,SdD,SdD,SDD,T2,fdag(T2, SDD),SDD,SdD,SdD,SDD,
	E2,E8,E4,conj(E6),E1,E7)
	return result
end

function square_ipeps_contraction_horizontal(T1, T2, SdD, SDD, env, HORIZONTAL_RULES, reinfo)
	nu,nl,nf,nd,nr = size(T1)
	χ = size(env[1])[1]

	E1 = symmetryreshape(env[1], χ,nl,nl,χ; reinfo = reinfo[1])[1]
	E2 = symmetryreshape(env[2], χ,nl,nl,χ; reinfo = reinfo[2])[1]
	E3 = symmetryreshape(env[3], χ,nl,nl,χ; reinfo = reinfo[2])[1]
	E4 = symmetryreshape(env[4], χ,nl,nl,χ; reinfo = reinfo[3])[1]
	E5 = symmetryreshape(env[5], χ,nl,nl,χ; reinfo = reinfo[2])[1]
	E6 = symmetryreshape(env[6], χ,nl,nl,χ; reinfo = reinfo[2])[1]
	result = HORIZONTAL_RULES(T1,SdD,fdag(T1, SDD),SdD,SDD,SDD,fdag(T2, SDD),SdD,SdD,SDD,T2,SDD,
	E1,E2,E3,E4,conj(E5),conj(E6))
	return result
end

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)

Initial `ipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel; Ni::Int, Nj::Int, folder = "./data/", symmetry = :none, atype = Array, D::Int, χ::Int, indD, indχ, dimsD, dimsχ, tol::Real, maxiter::Int, verbose = true)
    folder = folder*"/$(model)_$(Ni)x$(Nj)_$(indD)_$(dimsD)/"
    mkpath(folder)
    chkp_file = folder*"D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).jld2"
    if isfile(chkp_file)
        ipeps = load(chkp_file)["ipeps"]
        verbose && println("load iPEPS from $chkp_file")
    else
		symmetry == :none && (symmetry = :Z2)
		randdims = sum(prod.(
			zerosinitial(Val(symmetry), atype, ComplexF64, D, D, 4, D, D; 
						dir = [-1, -1, 1, 1, 1], 
						indqn = [indD, indD, getqrange(4)..., indD, indD],                    
						indims = [dimsD, dimsD, getblockdims(4)..., dimsD, dimsD], 
						q = [1]
						).dims))
        ipeps = randn(ComplexF64, randdims, Int(ceil(Ni*Nj/2)))
        verbose && println("random initial iPEPS $chkp_file")
    end 
	
    ipeps /= norm(ipeps)
	key = (folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter, indD, indχ, dimsD, dimsχ)
    return ipeps, key
end

function initial_consts(key)
	folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter, indD, indχ, dimsD, dimsχ = key
	SdD = U1swapgate(atype, ComplexF64, 4, D; 
		indqn = [getqrange(4)..., indD, getqrange(4)..., indD], 
		indims = [getblockdims(4)..., dimsD, getblockdims(4)..., dimsD]
	)
	SDD = U1swapgate(atype, ComplexF64, D, D; 
		indqn = [indD for _ in 1:4], 
		indims = [dimsD for _ in 1:4]
	)
    hx = reshape(atype{ComplexF64}(hamiltonian(model)), 4, 4, 4, 4)
	hy = reshape(atype{ComplexF64}(hamiltonian(model)), 4, 4, 4, 4)

	# SdD = asSymmetryArray(SdD, Val(symmetry); dir = [-1,-1,1,1], indqn = getqrange(4, D, 4, D), indims = getblockdims(4, D, 4, D))
	# SDD = asSymmetryArray(SDD, Val(symmetry); dir = [-1,-1,1,1], indqn = getqrange(D, D, D, D), indims = getblockdims(D, D, D, D))
	hx = asSymmetryArray(hx, Val(symmetry); dir = [-1,-1,1,1], indqn = getqrange(4, 4, 4, 4), indims = getblockdims(4, 4, 4, 4))
	hy = asSymmetryArray(hy, Val(symmetry); dir = [-1,-1,1,1], indqn = getqrange(4, 4, 4, 4), indims = getblockdims(4, 4, 4, 4))

	VERTICAL_RULES = generate_vertical_rules(D = D, χ = χ)
	HORIZONTAL_RULES = generate_horizontal_rules(D = D, χ = χ)

	reinfo = [[],[],[]]
	if symmetry == :U1
        indqn = [indχ, indD, indD, indχ]
        indims = [dimsχ, dimsD, dimsD, dimsχ]
		reinfo = [U1reshapeinfo((χ,D^2,χ), (χ,D,D,χ), [1,-1,1,-1], indqn, indims),
				  U1reshapeinfo((χ,D^2,χ), (χ,D,D,χ), [-1,-1,1,1], indqn, indims),
				  U1reshapeinfo((χ,D^2,χ), (χ,D,D,χ), [-1,1,-1,1], indqn, indims)]
	end

	SdD, SDD, hx, hy, HORIZONTAL_RULES, VERTICAL_RULES, reinfo
end

"""
    optimiseipeps(ipeps, h; χ, tol, maxiter, optimargs = (), optimmethod = LBFGS(m = 20))

return the tensor `bulk'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimiseipeps(ipeps::AbstractArray, key; f_tol = 1e-6, opiter = 100, verbose= false, optimmethod = LBFGS(m = 20))
    folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter, indD, indχ, dimsD, dimsχ = key
    consts = initial_consts(key)

    to = TimerOutput()
    f(x) = @timeit to "forward" double_ipeps_energy(atype(x), consts, key)
    ff(x) = double_ipeps_energy(atype(x), consts, key)
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

    folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter, indD, indχ, dimsD, dimsχ = key
    !(isdir(folder)) && mkdir(folder)
    if !(key === nothing)
        logfile = open(folder*"D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).log", "a")
        write(logfile, message)
        close(logfile)
        save(folder*"D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).jld2", "ipeps", os.metadata["x"])
    end
    return false
end