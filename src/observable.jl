using JLD2
using VUMPS:SquareVUMPSRuntime, ALCtoAC

function observable(model, atype, folder, D, χ, tol=1e-10, maxiter=10)
    Ni,Nj = 2,2
    observable_log = folder*"/$(model)_$(Ni)x$(Nj)/D$(D)_χ$(χ)_observable.log"
    # if isfile(observable_log)
    #     println("load observable from $(observable_log)")
    #     f = open(observable_log, "r" )
    #     occ,doubleocc = parse.(Float64,split(readline(f), "   "))
    #     close(f)
    # else
        ipeps, key = init_ipeps(model; Ni = Ni, Nj = Nj, atype = atype, folder = folder, D=D, χ=χ, tol=tol, maxiter= maxiter)
        folder, model, Ni, Nj, atype, D, χ, tol, maxiter = key
        T = reshape([parity_conserving(ipeps[:,:,:,:,:,ABBA(i)]) for i = 1:Ni*Nj], (Ni, Nj))
        b = reshape([zeros(D^2,D^2,D^2,D^2) for i = 1:Ni*Nj], (Ni, Nj))
        # b = reshape([permutedims(bulk(T[i]),(2,3,4,1)) for i = 1:Ni*Nj], (Ni, Nj))
        op = reshape([permutedims(bulkop(T[i]),(2,3,4,1,5,6)) for i = 1:Ni*Nj], (Ni, Nj))

        chkp_file_obs = folder*"/$(model)_$(Ni)x$(Nj)/obs_D$(D^2)_χ$(χ).jld2"
        FLo, FRo = load(chkp_file_obs)["env"]
        chkp_file_up = folder*"/$(model)_$(Ni)x$(Nj)/up_D$(D^2)_χ$(χ).jld2"                     
        rtup = SquareVUMPSRuntime(b, chkp_file_up, χ)   
        FLu, FRu, ALu, ARu, Cu = rtup.FL, rtup.FR, rtup.AL, rtup.AR, rtup.C
        chkp_file_down = folder*"/$(model)_$(Ni)x$(Nj)/down_D$(D^2)_χ$(χ).jld2"                              
        rtdown = SquareVUMPSRuntime(b, chkp_file_down, χ)   
        ALd,ARd,Cd = rtdown.AL,rtdown.AR,rtdown.C

        ACu = ALCtoAC(ALu,Cu)
        ACd = ALCtoAC(ALd,Cd)

        hocc = atype([0.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2])
        hdoubleocc = atype([0.0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1])
        Nzup = atype([0.0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1])
        Nzdn = atype([0.0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1])
        Nxup = atype([0.0 0 0 0; 0 1  1 0; 0  1 1 0; 0 0 0 2]./2)
        Nxdn = atype([0.0 0 0 0; 0 1 -1 0; 0 -1 1 0; 0 1 1 2]./2)
        Nyup = atype([0.0 0 0 0; 0 1 -1im 0; 0 1im 1 0; 0 0 0 2]./2)
        Nydn = atype([0.0 0 0 0; 0 1 1im 0; 0 -1im 1 0; 0 1 1 2]./2)
        occ = 0
        doubleocc = 0
        for j = 1:Nj, i = 1:Ni
            println("==========$i $j==========")
            ir = Ni + 1 - i
            ρ = ein"(((adf,abc),dgebpq),fgh),ceh -> pq"(FLo[i,j],ACu[i,j],op[i,j],ACd[ir,j],FRo[i,j])
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