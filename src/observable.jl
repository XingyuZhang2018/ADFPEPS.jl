using JLD2
using VUMPS:SquareVUMPSRuntime, ALCtoAC

function observable(model, Ni, Nj, atype, folder, symmetry, D, χ, indD, indχ, dimsD, dimsχ, tol=1e-10, maxiter=10)
    
    # if isfile(observable_log)
    #     println("load observable from $(observable_log)")
    #     f = open(observable_log, "r" )
    #     occ,doubleocc = parse.(Float64,split(readline(f), "   "))
    #     close(f)
    # else
        ipeps, key = init_ipeps(model; Ni=Ni, Nj=Nj, symmetry=symmetry, atype=atype, folder=folder, tol=tol, maxiter=maxiter, D=D, χ=χ, indD = indD, indχ = indχ, dimsD = dimsD, dimsχ = dimsχ)
        folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter, indD, indχ, dimsD, dimsχ = key
        observable_log = folder*"/D$(D)_χ$(χ)_observable.log"
        
        T = buildipeps(ipeps, key)
        SDD = U1swapgate(atype, ComplexF64, D, D; 
		indqn = [indD for _ in 1:4], 
		indims = [dimsD for _ in 1:4]
	    )
        M = reshape([bulk(T[i], SDD, indD, dimsD) for i = 1:Ni*Nj], (Ni, Nj))
        op = reshape([bulkop(T[i], SDD, indD, dimsD) for i = 1:Ni*Nj], (Ni, Nj))

        chkp_file_obs = folder*"/obs_D$(D^2)_χ$(χ).jld2"
        FLo, FRo = load(chkp_file_obs)["env"]
        chkp_file_up = folder*"/up_D$(D^2)_χ$(χ).jld2"                     
        rtup = SquareVUMPSRuntime(M, chkp_file_up, χ)   
        FLu, FRu, ALu, ARu, Cu = rtup.FL, rtup.FR, rtup.AL, rtup.AR, rtup.C
        chkp_file_down = folder*"/down_D$(D^2)_χ$(χ).jld2"                             
        rtdown = SquareVUMPSRuntime(M, chkp_file_down, χ)   
        ALd,ARd,Cd = rtdown.AL,rtdown.AR,rtdown.C

        ACu = ALCtoAC(ALu,Cu)
        ACd = ALCtoAC(ALd,Cd)

        hocc = atype([0.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2])
        hdoubleocc = atype([0.0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1])
        Nzup = atype([0.0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1])
        Nzdn = atype([0.0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1])
        Nxup = atype([0.0 0 0 0; 0 1  1 0; 0  1 1 0; 0 0 0 2]./2)
        Nxdn = atype([0.0 0 0 0; 0 1 -1 0; 0 -1 1 0; 0 0 0 2]./2)
        Nyup = atype([0.0 0 0 0; 0 1 -1im 0; 0 1im 1 0; 0 0 0 2]./2)
        Nydn = atype([0.0 0 0 0; 0 1 1im 0; 0 -1im 1 0; 0 0 0 2]./2)
        # U = atype([1 0 0 0;0 0 1 0;0 -1 0 0;0 0 0 1])
        # hocc, hdoubleocc, Nzup, Nzdn, Nxup, Nxdn, Nyup, Nydn = map(x->asSymmetryArray(x, Val(symmetry); dir = [-1,1], indqn = getqrange(size(x)...), indims = getblockdims(size(x)...)), [hocc, hdoubleocc, Nzup, Nzdn, Nxup, Nxdn, Nyup, Nydn])
        occ = 0
        doubleocc = 0
        for j = 1:Nj, i = 1:Ni
            println("==========$i $j==========")
            ir = Ni + 1 - i
            ρ = ein"(((adf,abc),dgebpq),fgh),ceh -> pq"(FLo[i,j],ACu[i,j],op[i,j],conj(ACd[ir,j]),FRo[i,j])
            # if (i,j) in [(2,1),(1,2)]
            #     ρ = U' * ρ * U
            # end
            ρ = asArray(ρ; indqn = getqrange(4, 4), indims = getblockdims(4, 4))
            Occ = ein"pq,pq -> "(ρ,hocc)
            DoubleOcc = ein"pq,pq -> "(ρ,hdoubleocc)
            NNzup = ein"pq,pq -> "(ρ,Nzup)
            NNzdn = ein"pq,pq -> "(ρ,Nzdn)
            NNxup = ein"pq,pq -> "(ρ,Nxup)
            NNxdn = ein"pq,pq -> "(ρ,Nxdn)
            NNyup = ein"pq,pq -> "(ρ,Nyup)
            NNydn = ein"pq,pq -> "(ρ,Nydn)
            n = ein"pp -> "(ρ) 
            occ += Array(Occ)[]/Array(n)[]
            doubleocc += Array(DoubleOcc)[]/Array(n)[]
            println("N = $(Array(Occ)[]/Array(n)[])")
            println("DN = $(Array(DoubleOcc)[]/Array(n)[])")
            println("Nz↑ = $(Array(NNzup)[]/Array(n)[])")
            println("Nz↓ = $(Array(NNzdn)[]/Array(n)[])")
            println("Nx↑ = $(Array(NNxup)[]/Array(n)[])")
            println("Nx↓ = $(Array(NNxdn)[]/Array(n)[])")
            println("Ny↑ = $(Array(NNyup)[]/Array(n)[])")
            println("Ny↓ = $(Array(NNydn)[]/Array(n)[])")
            println("{$(real(Array(NNzup)[]/Array(n)[])),$(real(Array(NNzdn)[]/Array(n)[])),$(real(Array(NNxup)[]/Array(n)[])),$(real(Array(NNxdn)[]/Array(n)[])),$(real(Array(NNyup)[]/Array(n)[])),$(real(Array(NNydn)[]/Array(n)[]))}")
        end

        occ = real(occ)/Ni/Nj
        doubleocc = real(doubleocc)/Ni/Nj
        message = "$occ   $doubleocc\n"
        logfile = open(observable_log, "a")
        write(logfile, message)
        close(logfile)
    # end
    return occ, doubleocc
end