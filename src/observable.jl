using JLD2
using TeneT:SquareVUMPSRuntime, ALCtoAC

function observable(model, Ni, Nj, atype, folder, symmetry, sitetype, d, D, χ, indD, indχ, dimsD, dimsχ, tol=1e-10, maxiter=10)
    
    # if isfile(observable_log)
    #     println("load observable from $(observable_log)")
    #     f = open(observable_log, "r" )
    #     occ,doubleocc = parse.(Float64,split(readline(f), "   "))
    #     close(f)
    # else

        ipeps, key = init_ipeps(model; Ni=Ni, Nj=Nj, 
        symmetry=symmetry, sitetype=sitetype, 
        atype=atype, folder=folder, tol=tol, maxiter=maxiter, miniter = 1, d=d, D=D, χ=χ, indD = indD, indχ = indχ, dimsD = dimsD, dimsχ = dimsχ)
        folder, model, Ni, Nj, symmetry, sitetype, atype, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ = key
        observable_log = folder*"/D$(D)_χ$(χ)_observable.log"
        
        T = buildipeps(ipeps, key)
        SDD = U1swapgate(atype, ComplexF64, D, D; 
						 indqn=[indD for _ in 1:4], 
						 indims=[dimsD for _ in 1:4],
						 ifZ2=sitetype.ifZ2
						)
        M = reshape([bulk(T[i], SDD, indD, dimsD) for i = 1:Ni*Nj], (Ni, Nj))
        op = reshape([bulkop(T[i], SDD, indD, dimsD, sitetype) for i = 1:Ni*Nj], (Ni, Nj))

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

        hocc1 = atype(zeros(9,9))
        for (i,j) in [[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]
            hocc1[i,j] = 1
        end
        hocc2 = atype(zeros(9,9))
        for (i,j) in [[2,2],[3,3],[5,5],[6,6],[8,8],[9,9]]
            hocc2[i,j] = 1
        end

        # hdoubleocc = atype([0.0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1])
        # Nzup = atype([0.0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1])
        # Nzdn = atype([0.0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1])
        # Nxup = atype([0.0 0 0 0; 0 1  1 0; 0  1 1 0; 0 0 0 2]./2)
        # Nxdn = atype([0.0 0 0 0; 0 1 -1 0; 0 -1 1 0; 0 0 0 2]./2)
        # Nyup = atype([0.0 0 0 0; 0 1 -1im 0; 0 1im 1 0; 0 0 0 2]./2)
        # Nydn = atype([0.0 0 0 0; 0 1 1im 0; 0 -1im 1 0; 0 0 0 2]./2)
        # U = atype([1 0 0 0;0 0 1 0;0 -1 0 0;0 0 0 1])
        # hocc, hdoubleocc, Nzup, Nzdn, Nxup, Nxdn, Nyup, Nydn = map(x->asSymmetryArray(x, Val(symmetry); dir = [-1,1], indqn = getqrange(size(x)...), indims = getblockdims(size(x)...)), [hocc, hdoubleocc, Nzup, Nzdn, Nxup, Nxdn, Nyup, Nydn])
        occ1 = 0
        occ2 = 0
        for j = 1:Nj, i = 1:Ni
            println("==========$i $j==========")
            ir = Ni + 1 - i
            ρ = ein"(((adf,abc),dgebpq),fgh),ceh -> pq"(FLo[i,j],ACu[i,j],op[i,j],conj(ACd[ir,j]),FRo[i,j])
            ρ = asArray(sitetype, ρ)
            n = ein"pp -> "(ρ)[]
            Occ1 = ein"pq,pq -> "(ρ,hocc1)[] / n 
            Occ2 = ein"pq,pq -> "(ρ,hocc2)[] / n
            
            occ1 += Occ1
            occ2 += Occ2
            println("Occ1 = $Occ1")
            println("Occ2 = $Occ2")
            println("Occ = $(Occ1+Occ2)")
        end

        occ1 = real(occ1)/Ni/Nj
        occ2 = real(occ2)/Ni/Nj
        occ  = real(occ)/Ni/Nj
        message = "$occ1    $occ2    $occ\n"
        logfile = open(observable_log, "a")
        write(logfile, message)
        close(logfile)
    # end
    return occ
end