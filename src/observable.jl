using JLD2
using TeneT:SquareVUMPSRuntime, ALCtoAC

function observable(ipeps, key; verbose = true)
    folder, model, Ni, Nj, symmetry, sitetype, atype, d, D, χ, tol, maxiter, miniter, qnD, qnχ, dimsD, dimsχ = key
    T = buildipeps(ipeps, key)
    SDD = U1swapgate(atype, ComplexF64, D, D; 
                        indqn=[qnD for _ in 1:4], 
                        indims=[dimsD for _ in 1:4],
                        ifZ2=sitetype.ifZ2
                    )
    M = reshape([bulk(T[i], SDD, qnD, dimsD) for i = 1:Ni*Nj], (Ni, Nj))
    op = reshape([bulkop(T[i], SDD, qnD, dimsD, sitetype) for i = 1:Ni*Nj], (Ni, Nj))

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

    hΔ = atype(zeros(9,9))
    for (i,j) in [[6,1]]
        hΔ[i,j] = 1/sqrt(2)
    end
    for (i,j) in [[8,1]]
        hΔ[i,j] = -1/sqrt(2)
    end
    # hocc1 = atype(zeros(3,3))
    # for (i,j) in [[2,2]]
    #     hocc1[i,j] = 1
    # end
    # hocc2 = atype(zeros(3,3))
    # for (i,j) in [[3,3]]
    #     hocc2[i,j] = 1
    # end

    occ1_acc = 0
    occ2_acc = 0
    Δ_acc = 0
    n_acc = 0
    for j = 1:Nj, i = 1:Ni
        verbose && println("==========$i $j==========")
        ir = Ni + 1 - i
        ρ = ein"(((adf,abc),dgebpq),fgh),ceh -> pq"(FLo[i,j],ACu[i,j],op[i,j],conj(ACd[ir,j]),FRo[i,j])
        ρ = asArray(sitetype, ρ)
        n = ein"pp -> "(ρ)[]
        occ1 = ein"pq,pq -> "(ρ,hocc1)[] / n 
        occ2 = ein"pq,pq -> "(ρ,hocc2)[] / n
        Δ = ein"pq,pq -> "(ρ,hΔ)[] / n
        
        n_acc += norm(n)
        occ1_acc += occ1
        occ2_acc += occ2
        Δ_acc += Δ
        if verbose
            println("norm(n) = $(norm(n))")
            println("occ1 = $occ1")
            println("occ2 = $occ2")
            println("occ = $((occ1+occ2)/2)")
            println("Δ = $Δ")
        end
    end
    n = abs(n_acc)/Ni/Nj
    occ1 = abs(occ1_acc)/Ni/Nj
    occ2 = abs(occ1_acc)/Ni/Nj
    occ  = (occ1+occ2)/2
    Δ = abs(Δ_acc)/Ni/Nj

    return n, occ, Δ
end