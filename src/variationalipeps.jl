using FileIO
using Optim, LineSearches
using LinearAlgebra: I, norm
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
	folder, model, Ni, Nj, symmetry, sitetype, atype, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ = key

	# TeneT
	# @show M[1,1]
	_, ALu, Cu, ARu, ALd, Cd, ARd, FLo, FRo, FLu, FRu = obs_env(M; 
																χ=χ, 
																maxiter=maxiter, 
																miniter=miniter, 
																tol = tol, 
																verbose=true, 
																savefile = true, 
																infolder=folder, 
																outfolder=folder, 
																updown = true, 
																downfromup = false, 
																show_every=Inf, 
																info = (indD, indχ, dimsD, dimsχ),
																savetol = 1e-5
																)

	ACu = reshape([ein"abc,cd->abd"(ALu[i],Cu[i]) for i = 1:Ni*Nj], (Ni, Nj))
	ACd = reshape([ein"abc,cd->abd"(ALd[i],Cd[i]) for i = 1:Ni*Nj], (Ni, Nj))
	return FLo, ACu, ARu, FRo, ARd, ACd, FLu, FRu
end

ABBA(i) = i in [1,4] ? 1 : 2

function buildipeps(ipeps, key)
	folder, model, Ni, Nj, symmetry, sitetype, atype, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ = key
	d = 9
	ipeps /= norm(ipeps)
	if symmetry == :none
		# info = Zygote.@ignore zerosinitial(Val(:Z2), atype, ComplexF64, D,D,3,D,D; dir = [-1,-1,1,1,1], q = [0])
		# reshape([asArray(Z2Array(info.parity, [reshape(atype(ipeps[1 + sum(prod.(info.dims[1:j-1])):sum(prod.(info.dims[1:j])), ABBA(i)]), tuple(info.dims[j]...)) for j in 1:length(info.dims)], info.size, info.dims, 1)) for i = 1:Ni*Nj], (Ni, Nj))
		reshape([ipeps[:,:,:,:,:,ABBA(i)] for i in 1:Ni*Nj],(Ni,Nj))
	else
		info = Zygote.@ignore zerosinitial(Val(symmetry), atype, ComplexF64, D,D,d,D,D; 
			dir = [-1,-1,1,1,1], 
			indqn = [indD, indD, getqrange(sitetype, d)..., indD, indD], 
			indims = [dimsD, dimsD, getblockdims(sitetype, d)..., dimsD, dimsD], 
			f=[0],
			ifZ2=sitetype.ifZ2
		)
		reshape([U1Array(info.qn, info.dir, atype(ipeps[:, ABBA(i)]), info.size, info.dims, 1, sitetype.ifZ2) for i = 1:Ni*Nj], (Ni, Nj))
	end
end

function double_ipeps_energy(ipeps::AbstractArray, consts, key)	
	folder, model, Ni, Nj, symmetry, sitetype, atype, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ = key
    SdD, SDD, h, HORIZONTAL_RULES, VERTICAL_RULES, ONSITE_RULES, reinfo = consts
	T = buildipeps(ipeps, key)
	M = reshape([bulk(T[i], SDD, indD, dimsD) for i = 1:Ni*Nj], (Ni, Nj))
	E1,E2,E3,E4,E5,E6,E7,E8 = ipeps_enviroment(M, key)

	etol = 0
	for j = 1:Nj, i = 1:Ni
		println("==========$(i),$(j)==========")
		ir = Ni + 1 - i
		jr = j + 1 - (j==Nj) * Nj
		
		Tij, Tijr, Tirj = T[i,j], T[i,jr], T[ir,j]
		ex = (E1[i,j],E2[i,j],E3[i,jr],E4[i,jr],E5[ir,jr],E6[ir,j])
		ρx = square_ipeps_contraction_horizontal(Tij, Tijr, SdD, SDD, ex, HORIZONTAL_RULES, reinfo)
		# ρ1 = reshape(ρ,16,16)
		# @show norm(ρ1-ρ1')
        Ex = ein"ijkl,ijkl -> "(ρx,h[1])[]
		nx = dtr(ρx) # nx = ein"ijij -> "(ρx)
		etol += Ex/nx
		println("─ = $(Ex/nx)") 

        ey = (E1[ir,j],E2[i,j],E4[ir,j],E6[i,j],E7[i,j],E8[i,j])
		ρy = square_ipeps_contraction_vertical(Tij, Tirj, SdD, SDD, ey, VERTICAL_RULES, reinfo)
		# ρ1 = reshape(ρ,16,16)
		# @show norm(ρ1-ρ1')
        Ey = ein"ijkl,ijkl -> "(ρy,h[1])[]
		ny = dtr(ρy) # ny = ein"ijij -> "(ρy)[]
		etol += Ey/ny
		println("│ = $(Ey/ny)")

		eo = (E1[i,j],E2[i,j],E4[i,j],E6[ir,j])
		ρo = square_ipeps_contraction_onsite(Tij, SDD, eo, ONSITE_RULES, reinfo)
		# ρ1 = reshape(ρ,16,16)
		# @show norm(ρ1-ρ1')
		Eo = ein"ij,ij -> "(ρo,h[2])[]
		no = tr(ρo)
		etol += Eo/no
		println("o = $(Eo/no)")
	end
	@show etol/Ni/Nj
	return real(etol)/Ni/Nj
end

function nodiffenv(ipeps::AbstractArray, consts, key)	
	folder, model, Ni, Nj, symmetry, sitetype, atype, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ = key
    SdD, SDD, h, HORIZONTAL_RULES, VERTICAL_RULES, ONSITE_RULES, reinfo = consts
	T = buildipeps(ipeps, key)
	M = reshape([bulk(T[i], SDD, indD, dimsD) for i = 1:Ni*Nj], (Ni, Nj))
	ipeps_enviroment(M, key)
	nothing
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

function square_ipeps_contraction_onsite(T, SDD, env, ONSITE_RULES, reinfo)
	nu,nl,nf,nd,nr = size(T)
	χ = size(env[1])[1]

	E1 = symmetryreshape(env[1], χ,nl,nl,χ; reinfo = reinfo[1])[1]
	E2 = symmetryreshape(env[2], χ,nl,nl,χ; reinfo = reinfo[2])[1]
	E4 = symmetryreshape(env[3], χ,nl,nl,χ; reinfo = reinfo[3])[1]
	E6 = symmetryreshape(env[4], χ,nl,nl,χ; reinfo = reinfo[2])[1]
	result = ONSITE_RULES(T,fdag(T, SDD),SDD,SDD,E1,E2,E4,conj(E6))
	return result
end

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)

Initial `ipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel; 
					Ni::Int, Nj::Int, 
					folder = "./data/", 
					symmetry = :U1, 
					sitetype = tJZ2(),
					atype = Array, 
					d::Int, D::Int, χ::Int, 
					indD, indχ, dimsD, dimsχ, 
					tol::Real, maxiter::Int, miniter::Int, verbose = true
					)
	folder = folder*"/$(model)_$(Ni)x$(Nj)_$(indD)_$(dimsD)/"
    mkpath(folder)
    chkp_file = folder*"D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).jld2"
    if isfile(chkp_file)
        ipeps = load(chkp_file)["ipeps"]
        verbose && println("load iPEPS from $chkp_file")
    else
		if symmetry == :none
			ipeps = randn(ComplexF64, D, D, d, D, D, Int(ceil(Ni*Nj/2)))
		else
			randdims = sum(prod.(
				zerosinitial(Val(:U1), atype, ComplexF64, D, D, d, D, D; 
							dir = [-1, -1, 1, 1, 1], 
							indqn = [indD, indD, getqrange(sitetype, d)..., indD, indD],                    
							indims = [dimsD, dimsD, getblockdims(sitetype, d)..., dimsD, dimsD], 
							f=[0],
							ifZ2=sitetype.ifZ2
							).dims))
			ipeps = randn(ComplexF64, randdims, Int(ceil(Ni*Nj/2)))
		end
        verbose && println("random initial iPEPS $chkp_file")
    end 
	println("parameters: $(prod(size(ipeps))/Int(ceil(Ni*Nj/2)))")
    ipeps /= norm(ipeps)
	key = (folder, model, Ni, Nj, symmetry, sitetype, atype, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ)
    return ipeps, key
end

function initial_consts(key)
	folder, model, Ni, Nj, symmetry, sitetype, atype, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ = key

	h = atype{ComplexF64}.(hamiltonian(model))
	h = (asSymmetryArray(h[1], Val(symmetry), sitetype; dir = [-1,-1,1,1]),
	     asSymmetryArray(h[2], Val(symmetry), sitetype; dir = [-1,1]))

	d = size(h[1], 1)
	# h = atype{ComplexF64}(hamiltonian(model))
	# h = asSymmetryArray(h, Val(symmetry); dir = [-1,-1,1,1])
	# d = size(h, 1)
	
	if symmetry == :none
		SdD = swapgate(sitetype, atype, ComplexF64, d, D)
		SDD = swapgate(sitetype, atype, ComplexF64, D, D)
	else
		SdD = U1swapgate(atype, ComplexF64, d, D; 
						 indqn = [getqrange(sitetype, d)..., indD, getqrange(sitetype, d)..., indD], 
						 indims = [getblockdims(sitetype, d)..., dimsD, getblockdims(sitetype, d)..., dimsD],
						 ifZ2=sitetype.ifZ2
						)
		SDD = U1swapgate(atype, ComplexF64, D, D; 
						 indqn = [indD for _ in 1:4], 
						 indims = [dimsD for _ in 1:4],
						 ifZ2=sitetype.ifZ2
						)
	end

	VERTICAL_RULES = generate_vertical_rules(D = D, χ = χ)
	HORIZONTAL_RULES = generate_horizontal_rules(D = D, χ = χ)
	ONSITE_RULES = generate_onsite_rules(D = D, χ = χ)

	reinfo = [[],[],[]]
	if symmetry != :none
        indqn = [indχ, indD, indD, indχ]
        indims = [dimsχ, dimsD, dimsD, dimsχ]
		reinfo = [U1reshapeinfo((χ,D^2,χ), (χ,D,D,χ), [1,-1,1,-1], indqn, indims, sitetype.ifZ2),
				  U1reshapeinfo((χ,D^2,χ), (χ,D,D,χ), [-1,-1,1,1], indqn, indims, sitetype.ifZ2),
				  U1reshapeinfo((χ,D^2,χ), (χ,D,D,χ), [-1,1,-1,1], indqn, indims, sitetype.ifZ2)]
	end

	SdD, SDD, h, HORIZONTAL_RULES, VERTICAL_RULES, ONSITE_RULES, reinfo
end

"""
    optimiseipeps(ipeps, h; χ, tol, maxiter, optimargs = (), optimmethod = LBFGS(m = 20))

return the tensor `bulk'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimiseipeps(ipeps::AbstractArray, key; 
						f_tol = 1e-6, opiter = 100, 
						maxiter_ad = 10, miniter_ad = 1,
						verbose= false, 
						optimmethod = LBFGS(m = 20,
						alphaguess=LineSearches.InitialStatic(alpha=1e-5,scaled=true)))

    folder, model, Ni, Nj, symmetry, sitetype, atype, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ = key
    consts = initial_consts(key)

	keyback = folder, model, Ni, Nj, symmetry, sitetype, atype, D, χ, tol, maxiter_ad, miniter_ad, indD, indχ, dimsD, dimsχ

    f(x) = double_ipeps_energy(atype(x), consts, key)
	ff(x) = double_ipeps_energy(atype(x), consts, keyback)
	function g(x)
        println("for backward convergence:")
        f(x)
        println("true backward:")
        grad = Zygote.gradient(ff,atype(x))[1]
		# gnorm = norm(grad) 
		# gnorm> 1e-1 && (grad /= gnorm)
        return grad
    end
    res = optimize(f, g, 
        ipeps, optimmethod,inplace = false,
        Optim.Options(f_tol=f_tol, iterations=opiter,
        extended_trace=true,
        callback=os->writelog(os, key)),
        )
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

    folder, model, Ni, Nj, symmetry, sitetype, atype, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ = key
    !(isdir(folder)) && mkdir(folder)
    if !(key === nothing)
        logfile = open(folder*"D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).log", "a")
        write(logfile, message)
        close(logfile)
        save(folder*"D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).jld2", "ipeps", os.metadata["x"])
    end
    return false
end