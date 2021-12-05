using JLD2
using VUMPS:SquareVUMPSRuntime
using LinearAlgebra

function observable(model, Ni, Nj, atype, folder, D, χ, tol=1e-10, maxiter=10)
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

        hocc = atype([0.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2])
        hdoubleocc = atype([0.0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1])
        hNzup = atype([0.0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1])
        hNzdn = atype([0.0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1])
        hNxup = atype([0.0 0 0 0; 0 1  1 0; 0  1 1 0; 0 0 0 2]./2)
        hNxdn = atype([0.0 0 0 0; 0 1 -1 0; 0 -1 1 0; 0 1 1 2]./2)
        hNyup = atype([0.0 0 0 0; 0 1 -1im 0; 0 1im 1 0; 0 0 0 2]./2)
        hNydn = atype([0.0 0 0 0; 0 1 1im 0; 0 -1im 1 0; 0 1 1 2]./2)
        hx = reshape(atype(hamiltonian(Hubbard(model.t,model.U,model.μ))), 4, 4, 4, 4)
        hy = reshape(atype(hamiltonian(Hubbard(model.t,model.U,model.μ))), 4, 4, 4, 4)
        # hx = reshape(atype(hamiltonian(model)), 4, 4, 4, 4)
        # hy = reshape(atype(hamiltonian(model)), 4, 4, 4, 4)
        # UniformT = diagm(ones(16))
        # for i in [7,8,12]
        #     UniformT[i,i] = -1.0
        # end
        UniformT = [1 0 0 0;0 0 1 0;0 1 0 0;0 0 0 -1]
        UniformT = reshape(ein"ab,cd -> acbd"(I(4),UniformT),16,16)
        UniformT = reshape(UniformT,4,4,4,4)
        occ1,occ2 = 0,0
        doubleocc1,doubleocc2 = 0,0
        Ex,Ey = 0,0
        function opobser(ρ,op,n)
            op1 = ein"ij,kl -> ikjl"(op,I(4))
            op2 = ein"ij,kl -> ikjl"(I(4),op)
            obs1 = Array(ein"ijkl,ijkl -> "(ρ,op1))[]/Array(n)[]
            obs2 = Array(ein"ijkl,ijkl -> "(ρ,op2))[]/Array(n)[]
            return obs1,obs2
        end 

        for j = 1:Nj, i = 1:Ni
            println("==========$(i) $(j)==========")
            ir = Ni + 1 - i
            jr = j + 1 - (j==Nj) * Nj
            ex = (E1[i,j],E2[i,j],E3[i,jr],E4[i,jr],E5[ir,jr],E6[ir,j],E7[i,j],E8[i,j])
            ρx = square_ipeps_contraction_horizontal(T[i,j],T[i,jr],ex)
            ρx = ein"abcd,cdef,efgh -> abgh"(UniformT, ρx, UniformT)
            nx = ein"ijij -> "(ρx) 
            Occ = opobser(ρx,hocc,nx)
            occ1 += Occ[1]
            occ2 += Occ[2]
            DoubleOcc = opobser(ρx,hdoubleocc,nx)
            doubleocc1 += DoubleOcc[1]
            doubleocc2 += DoubleOcc[2]
            Nzup1,Nzup2 = opobser(ρx,hNzup,nx)
            Nzdn1,Nzdn2 = opobser(ρx,hNzdn,nx)
            Nxup1,Nxup2 = opobser(ρx,hNxup,nx)
            Nxdn1,Nxdn2 = opobser(ρx,hNxdn,nx)
            Nyup1,Nyup2 = opobser(ρx,hNyup,nx)
            Nydn1,Nydn2 = opobser(ρx,hNydn,nx)
            Ex += Array(ein"ijkl,ijkl -> "(ρx,hx))[]/Array(nx)[]
            # println("N1 = $(Occ[1])")
            # println("N2 = $(Occ[2])")
            # println("DN1 = $(DoubleOcc[1])")
            # println("DN2 = $(DoubleOcc[2])")
            println("z↑1 = $(Nzup1)")
            println("z↑2 = $(Nzup2)")
            # println("z↓1 = $(Nzdn1)")
            # println("z↓2 = $(Nzdn2)")
            # println("x↑1 = $(Nxup1)")
            # println("x↑2 = $(Nxup2)")
            # println("x↓1 = $(Nxdn1)")
            # println("x↓2 = $(Nxdn2)")
            # println("y↑1 = $(Nyup1)")
            # println("y↑2 = $(Nyup2)")
            # println("y↓1 = $(Nydn1)")
            # println("y↓2 = $(Nydn2)")

            ey = (E1[ir,j],E2[i,j],E3[i,j],E4[ir,j],E5[ir,jr],E6[i,j],E7[i,j],E8[i,j])
            ρy = square_ipeps_contraction_vertical(T[i,j],T[ir,j],ey)
            ρy = ein"abcd,cdef,efgh -> abgh"(UniformT, ρy, UniformT)
            ny = ein"ijij -> "(ρy)
            Occ = opobser(ρy,hocc,ny)
            occ1 += Occ[1]
            occ2 += Occ[2]
            DoubleOcc = opobser(ρy,hdoubleocc,ny)
            doubleocc1 += DoubleOcc[1]
            doubleocc2 += DoubleOcc[2]
            Nzup1,Nzup2 = opobser(ρy,hNzup,ny)
            Nzdn1,Nzdn2 = opobser(ρy,hNzdn,ny)
            Nxup1,Nxup2 = opobser(ρy,hNxup,ny)
            Nxdn1,Nxdn2 = opobser(ρy,hNxdn,ny)
            Nyup1,Nyup2 = opobser(ρy,hNyup,ny)
            Nydn1,Nydn2 = opobser(ρy,hNydn,ny)
            Ey += Array(ein"ijkl,ijkl -> "(ρy,hy))[]/Array(ny)[]
            # println("N1 = $(Occ[1])")
            # println("N2 = $(Occ[2])")
            # println("DN1 = $(DoubleOcc[1])")
            # println("DN2 = $(DoubleOcc[2])")
            println("z↑1 = $(Nzup1)")
            println("z↑2 = $(Nzup2)")
            # println("z↓1 = $(Nzdn1)")
            # println("z↓2 = $(Nzdn2)")
            # println("x↑1 = $(Nxup1)")
            # println("x↑2 = $(Nxup2)")
            # println("x↓1 = $(Nxdn1)")
            # println("x↓2 = $(Nxdn2)")
            # println("y↑1 = $(Nyup1)")
            # println("y↑2 = $(Nyup2)")
            # println("y↓1 = $(Nydn1)")
            # println("y↓2 = $(Nydn2)")
        end

        occ = real(occ1 + occ2)/4/Ni/Nj
        doubleocc = real(doubleocc1 + doubleocc2)/4/Ni/Nj
        E = real(Ex + Ey)/Ni/Nj
        message = "$occ   $doubleocc\n"
        logfile = open(observable_log, "w")
        write(logfile, message)
        close(logfile)
    # end
    return occ, doubleocc, E
end