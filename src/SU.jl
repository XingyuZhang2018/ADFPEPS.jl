using TeneT: qrpos, svd!, invDiagU1Matrix
using LinearAlgebra

@with_kw struct SU <: Algorithm
    dτ::Float64 = 0.4
    tratio::Float64 = 0.7
    Mindτ::Float64 = 0.0001
    NoUp::Int64 = 1000
    doEstimate::Bool = true
    tolerance_Es::Float64 = 1.0e-8
    count_upper::Int64 = 200
    count_lower::Int64 = 100
end

function initλΓ(ST, D, d)
    λ = [Iinitial(ST, D; dir = [-1,1]) for _ in 1:4]
    Γ = [randinitial(ST, d, D, D, D, D; dir = [1, -1, -1, 1, 1]) for _ in 1:2]
    normalize!(Γ)
    return λ, Γ
end 

order(a) = a<5 ? a : (a-4)

function qndims(ΓA, ind)     
    indqn_D = Int64[]
    indims_D = Int64[]
    qn = ΓA.qn 
    dims = ΓA.dims 
    for i in 1:length(ΓA.qn)
        if qn[i][ind] in indqn_D
            continue
        end 
        push!(indqn_D, qn[i][ind]) 
        push!(indims_D, dims[i][ind]) 
    end   
    return indqn_D, indims_D
end

function evoGate(ST, model, dτ)
    h = ST.atype{ST.dtype}.(hamiltonian(model))
    d = size(h[1], 1)
    
    U_2sites = reshape(exp(-0.5 * dτ * reshape(h[1], d^2,d^2)), d,d,d,d)
    U_2sites = asSymmetryArray(U_2sites, ST.symmetry, ST.stype; dir = [-1,-1,1,1]) 

    U_local = exp(-0.5 * dτ * h[2]) 
    U_local = asSymmetryArray(U_local, ST.symmetry, ST.stype; dir = [-1,1])   
    
    return U_2sites, U_local
end

# fig form https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.165104
function update_row!(ST, Γ, λ, Udτ, D_truc; whichbond)
    sitetype, atype, dtype = ST.stype, ST.atype, ST.dtype
    # up: left=1, lm=1
    # down: left=0, lm=3
    if whichbond == "right"
        left = 1
        right = 2
        lm = 1
    elseif whichbond == "left"
        left = 2
        right = 1
        lm = 3
    else
        println("Wrong Position in update row!")
    end
    
    ΓA = Γ[left]
    ΓB = Γ[right]

    D = ΓA.size[4]
    d = Udτ.size[1]

    indqn_D, indims_D = qndims(ΓA, 4) 
    indqn_d, indims_d = getqrange(sitetype, d), getblockdims(sitetype, d)

    swap_dD = U1swapgate(atype, dtype, d, D; 
                        indqn = [indqn_d..., indqn_D, indqn_d..., indqn_D], 
                        indims = [indims_d..., indims_D, indims_d..., indims_D],
                        ifZ2=sitetype.ifZ2
                        ) 
    # step 1: Fig.29(a)
    ΓAA = ein"pludr,pdsy -> sluyr"(ΓA, swap_dD) 

    # step 2, Fig.34(a): obtain left part Θ  clockwise: dlurs 
    Γ_0 = ein"((sludr,dx),yl),zu -> xyzrs"(ΓAA, λ[order(lm+1)], λ[order(lm+2)], λ[order(lm+3)])
 
    # step 2': Fig.29(c) SVD/QR with no truncation 
    Γ_0 = reshape(Γ_0, D^3, D*d) 
    Q, R = qrpos(Γ_0)
    
    WA = reshape(Q, D, D, D, D*d) 
    MA = reshape(R, D*d, D, d)

    # step 3, Fig.34(a): right part Γ clock-wise   slurd
    Γ_1 = ein"((pludr,xu),ry),dz -> plxyz"(ΓB, λ[order(lm+1)], λ[order(lm+2)], λ[order(lm+3)])

    # step 3': Fig.29(d) SVD/RQ with no truncation      
    Q, R = qrpos(reshape(Γ_1, d*D, D^3)')
    UB = reshape(Q', D*d, D, D, D)
    NB = reshape(R', d, D, D*d)

    # step 4, Fig.34(a) or Towards Fig B.1(II)
    Θ0 = ein"(lrs,rx),pxy -> lspy"(MA, λ[lm], NB) # anticlock-wise,  
  
    Θ = ein"lspr,psqt -> ltqr"(Θ0, Udτ) # anticlock-wise,
     
    shape = Θ.size
    temp = reshape(Θ, shape[2]*shape[1], shape[3]*shape[4]) #  
    U, S, V = svd!(temp; trunc=D_truc, middledir=1)  #  

    MA_new = reshape(U, shape[1], shape[2], D_truc)
    NB_new = reshape(V', D_truc, shape[3], shape[4])

    S = Diagonal(S) 

    # step 5, Fig.(35)
    inv_λ2 = invDiagU1Matrix(λ[order(lm+1)])
    inv_λ3 = invDiagU1Matrix(λ[order(lm+2)])
    inv_λ4 = invDiagU1Matrix(λ[order(lm+3)])
 
    #  WA:(d,l,u,r)  # MA_new:(lpr)
    ΓAA_new = ein"(((dlur,dx),yl),zu),rpw -> pyzxw"(WA, inv_λ2, inv_λ3, inv_λ4, MA_new)  
    # NB_new:(lpr)  # UB:(l,u,r,d )
    ΓB_new = ein"(((mpl,lurd),xu),ry),dz -> pmxzy"(NB_new, UB, inv_λ2, inv_λ3, inv_λ4)
 
  
    # step 6, Fig.(32b)
    indqn_D, indims_D = qndims(ΓAA_new, 4)  
    swap_dD = U1swapgate(atype, dtype, d, D; 
                        indqn = [indqn_d..., indqn_D, indqn_d..., indqn_D], 
                        indims = [indims_d..., indims_D, indims_d..., indims_D],
                        ifZ2=sitetype.ifZ2
                        ) 
    ΓA_new = ein"pludr,pdsy -> sluyr"(ΓAA_new, swap_dD) 

 
    Γ[left] = ΓA_new
    Γ[right] = ΓB_new
 
    Snorm = norm(S)
    S /= Snorm

    λ[lm] = S
      
    return Snorm
end

function update_column!(ST, Γ, λ, Udτ, D_truc; whichbond)
    sitetype, atype, dtype = ST.stype, ST.atype, ST.dtype
    if whichbond =="down"
        left = 1
        right = 2
        lm = 2
    elseif whichbond =="up"
        left = 2
        right = 1
        lm = 4
    else
        println("Wrong Position in update column!")
    end 

    ΓA = Γ[left]
    ΓB = Γ[right]

    D = ΓA.size[2]
    d = Udτ.size[1]
    
    # d, D, d, D
    indqn_D, indims_D = qndims(ΓB, 2)  
    indqn_d, indims_d = getqrange(sitetype, d), getblockdims(sitetype, d)

    swap_dD = U1swapgate(atype, dtype, d, D; 
                        indqn = [indqn_d..., indqn_D, indqn_d..., indqn_D], 
                        indims = [indims_d..., indims_D, indims_d..., indims_D],
                        ifZ2=sitetype.ifZ2
                        )   	 
    # step 1: Fig.29(a)
    ΓBB = ein"pludr,pxsl -> sxudr"(ΓB, swap_dD) 

    # step 2, Fig.34(a): obtain Θ up  clockwise  [lurds]
    Γ_0 = ein"((sludr,xl),yu),rz -> xyzds"(ΓA, λ[order(lm+1)], λ[order(lm+2)], λ[order(lm+3)])
  
    # step 2': Fig.29(c) SVD/QR with no truncation 
    Γ_0 = reshape(Γ_0, D^3, D*d) # 
    Q, R = qrpos(Γ_0)
    
    WA = reshape(Q, D, D, D, D*d) 
    MA = reshape(R, D*d, D, d)

    # step 3, Fig.34(a): clock-wise  down: [surdl] 
    Γ_1 = ein"((pludr,rx),dy),zl -> puxyz"(ΓBB, λ[order(lm+1)], λ[order(lm+2)], λ[order(lm+3)])

    # step 3': Fig.29(d) SVD/RQ with no truncation      
    Q, R = qrpos(reshape(Γ_1, d*D, D^3)')
    UB = reshape(Q', D*d, D, D, D)
    NB = reshape(R', d, D, D*d)

    # step 4, Fig.34(a) or Towards Fig B.1(II)
    Θ0 = ein"(udp,dy),syx -> upsx"(MA, λ[lm], NB) # anticlock-wise, left 
    Θ = ein"upsd,psxy -> uxyd"(Θ0, Udτ) # anticlock-wise 
 
    shape = Θ.size
    temp = reshape(Θ, shape[2]*shape[1], shape[3]*shape[4]) #  
    U, S, V = svd!(temp; trunc=D_truc, middledir=1)  #  
    MA_new = reshape(U, shape[1], shape[2], D_truc)
    NB_new = reshape(V', D_truc, shape[3], shape[4])    
    S = Diagonal(S)
    

    # step 5, Fig.(35)
    inv_λ2 = invDiagU1Matrix(λ[order(lm+1)])
    inv_λ3 = invDiagU1Matrix(λ[order(lm+2)])
    inv_λ4 = invDiagU1Matrix(λ[order(lm+3)])

    #  WA:(l,u,r,d  ),  l3, l4, l1
    ΓA_new = ein"(((lurd,xl),yu),rz),dpw -> pxywz"(WA, inv_λ2, inv_λ3, inv_λ4, MA_new)  
    # UB:(u,r,d,l)
    ΓBB_new = ein"(((mpu,urdl),rx),dy),zl -> pzmyx"(NB_new, UB, inv_λ2, inv_λ3, inv_λ4)

    # step 6, Fig.(32b)
    indqn_D, indims_D = qndims(ΓBB_new, 2)  
    swap_dD = U1swapgate(atype, dtype, d, D; 
                        indqn = [indqn_d..., indqn_D, indqn_d..., indqn_D], 
                        indims = [indims_d..., indims_D, indims_d..., indims_D],
                        ifZ2=sitetype.ifZ2
                        )   
    ΓB_new = ein"pludr,pxsl -> sxudr"(ΓBB_new, swap_dD) 

    Γ[left] = ΓA_new
    Γ[right] = ΓB_new

    Snorm = norm(S)
    S /= Snorm
 
    λ[lm] = S
      
    return  Snorm 
end 