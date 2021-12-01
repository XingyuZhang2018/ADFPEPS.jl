using JLD2
using VUMPS:SquareVUMPSRuntime

function observable(model, atype, folder, D, χ, tol=1e-10, maxiter=10)
    Ni,Nj = 2,2
    observable_log = folder*"/$(model)_$(Ni)x$(Nj)/D$(D)_χ$(χ)_observable.log"
    if isfile(observable_log)
        println("load observable from $(observable_log)")
        f = open(observable_log, "r" )
        occ,doubleocc = parse.(Float64,split(readline(f), "   "))
        close(f)
    else
        ipeps, key = init_ipeps(model; Ni = Ni, Nj = Nj, atype = atype, folder = folder, D=D, χ=χ, tol=tol, maxiter= maxiter)
        folder, model, Ni, Nj, atype, D, χ, tol, maxiter = key
        T = reshape([parity_conserving(ipeps[:,:,:,:,:,ABBA(i)]) for i = 1:Ni*Nj], (Ni, Nj))
        b = reshape([permutedims(bulk(T[i]),(2,3,4,1)) for i = 1:Ni*Nj], (Ni, Nj))
        
        chkp_file_obs = folder*"/$(model)_$(Ni)x$(Nj)/obs_D$(D^2)_χ$(χ).jld2"
        FLo, FRo = load(chkp_file_obs)["env"]
        chkp_file_up = folder*"/$(model)_$(Ni)x$(Nj)/up_D$(D^2)_χ$(χ).jld2"                     
        rtup = SquareVUMPSRuntime(b, chkp_file_up, χ; verbose = false)   
        FLu, FRu, ALu, ARu, Cu = rtup.FL, rtup.FR, rtup.AL, rtup.AR, rtup.C
        chkp_file_down = folder*"/$(model)_$(Ni)x$(Nj)/down_D$(D^2)_χ$(χ).jld2"                              
        rtdown = SquareVUMPSRuntime(b, chkp_file_down, χ; verbose = false)   
        ALd,ARd,Cd = rtdown.AL,rtdown.AR,rtdown.C

        E1 = FLo
        E2 = reshape([ein"abc,cd->abd"(ALu[i],Cu[i]) for i = 1:Ni*Nj], (Ni, Nj))
        E3 = ARu
        E4 = FRo
        E5 = ARd
        E6 = reshape([ein"abc,cd->abd"(ALd[i],Cd[i]) for i = 1:Ni*Nj], (Ni, Nj))
        E7 = FLu
        E8 = FRu

        hocc = reshape(atype(hamiltonian(Occupation())), 4, 4, 4, 4)
        hdoubleocc = reshape(atype(hamiltonian(DoubleOccupation())), 4, 4, 4, 4)
        occ = 0
        doubleocc = 0
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            jr = j + 1 - (j==Nj) * Nj
            ex = (E1[i,j],E2[i,j],E3[i,jr],E4[i,jr],E5[ir,jr],E6[ir,j],E7[i,j],E8[i,j])
            ρx = square_ipeps_contraction_horizontal(T[i,j],T[i,jr],ex)
            Occ = ein"ijkl,ijkl -> "(ρx,hocc)
            DoubleOcc = ein"ijkl,ijkl -> "(ρx,hdoubleocc)
            nx = ein"ijij -> "(ρx) 
            occ += Array(Occ)[]/Array(nx)[]
            doubleocc += Array(DoubleOcc)[]/Array(nx)[]
            println("N = $(Array(Occ)[]/Array(nx)[])")
            println("DN = $(Array(DoubleOcc)[]/Array(nx)[])")
    
            ey = (E1[ir,j],E2[i,j],E3[i,j],E4[ir,j],E5[ir,jr],E6[i,j],E7[i,j],E8[i,j])
            ρy = square_ipeps_contraction_vertical(T[i,j],T[ir,j],ey)
            ny = ein"ijij -> "(ρy)
            Occ = ein"ijkl,ijkl -> "(ρy,hocc)
            DoubleOcc = ein"ijkl,ijkl -> "(ρy,hdoubleocc)
            occ += Array(Occ)[]/Array(ny)[]
            doubleocc += Array(DoubleOcc)[]/Array(ny)[]
            println("N = $(Array(Occ)[]/Array(ny)[])")
            println("DN = $(Array(DoubleOcc)[]/Array(ny)[])")
        end

        occ = real(occ)/Ni/Nj
        doubleocc = real(doubleocc)/Ni/Nj
        message = "$occ   $doubleocc\n"
        logfile = open(observable_log, "a")
        write(logfile, message)
        close(logfile)
    end
    return occ, doubleocc
end