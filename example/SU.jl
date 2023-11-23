using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules
using CUDA
using Random
using TeneT
using TeneT: randU1, zerosU1, IU1, qrpos, lqpos, sysvd!, initialA, zerosinitial
using LinearAlgebra
using OMEinsum

atype = Array 
dtype = ComplexF64
D = 2
d = 9
sitetype = tJZ2()
model = tJ_bilayer(3.0,1.0,0.0,2.0,0.0)

# ================== initialize Lambda and gamma ======================
lambda = []
for i = 1:5    
    push!(lambda, IU1(sitetype, atype, dtype, D; dir = [-1,1]))
end

Gamma = []
for i = 1:2
    temp = randU1(sitetype, atype, dtype, d, D, D, D, D ; dir = [-1, -1, 1, 1, 1])
    push!(Gamma, temp)
end 

h = atype{ComplexF64}.(hamiltonian(model))


# ================   SU  =================
dtau = 0.2
nsteps = 100
order = 2

# ------------------------  gate -------------------
U_local = exp(-0.5*dtau*h[2]  ) 
function gen_U_2sites( h2, dtau )
    d = size(h2)[1]
    temp = reshape(h2, d^2, d^2)
    temp = exp( -1.0 * dtau * temp ) 
    temp = reshape(temp, d, d, d, d)
    return temp
end
U_2sites_1 = gen_U_2sites(h[2], 0.5*dtau )
 
U_2sites_1 = asU1Array(sitetype, U_2sites_1; dir = [-1,-1,1,1])
U_local = asU1Array(sitetype, U_2sites; dir = [-1,1])


normalzie(A::U1Array) = A/norm(A)

# --------------------- update once --------------------
function update_once_2nd(Gamma, lambda, U_local, U_2sites_1, doEstimate)
    Gamma[1] = ein"pludr,ps -> sludr"(Gamma[1], U_local)
    Gamma[2] = ein"pludr,ps -> sludr"(Gamma[2], U_local) 

    if not doEstimate
        # step 5, may not needed
        normalize( Gamma[1] )
        normalize( Gamma[2] )   
    end  
    
    temp = 1.0
    # down
    temp *= update_row(Gamma, lambda, U_2sites_1, position="up")
    temp *= update_column(Gamma, lambda, U_2sites_1, position="right") 
    temp *= update_row(Gamma, lambda, U_2sites_1, position="down")    
    
    temp *= update_column(Gamma, lambda, U_2sites_1, position="left")
    temp *= update_column(Gamma, lambda, U_2sites_1, position="left")
    
    temp *= update_row(Gamma, lambda, U_2sites_1, position="down")
    temp *= update_column(Gamma, lambda, U_2sites_1, position="right")
    temp *= update_row(Gamma, lambda, U_2sites_1, position="up")    
    
    
    Gamma[0] = ein"pludr,ps -> sludr"(Gamma[1], U_local)
    Gamma[1] = ein"pludr,ps -> sludr"(Gamma[2], U_local)
    
    if not doEstimate
        # step 5, may not needed
        normalize( Gamma[1] )
        normalize( Gamma[2] )
    end
    
    return Gamma, lamda, temp
    
end
 

# only for U1 Matrix
function cutvd!(A::U1Array{T,N}) where {T,N}
    qn = A.qn
    div = A.division
    atype = _arraytype(A.tensor)

    Adims = A.dims
    Abdiv = blockdiv(Adims)
    tensor = [reshape(@view(A.tensor[Abdiv[i]]), prod(Adims[i][1:div]), prod(Adims[i][div+1:end])) for i in 1:length(Abdiv)]

    Utensor = Vector{atype{T}}()
    Stensor = Vector{atype{T}}()
    Vtensor = Vector{atype{T}}()
    svals = []
    @inbounds @simd for t in tensor
        U, S, V = sysvd!(t)
        push!(Utensor, U)
        push!(Stensor, S)
        push!(Vtensor, V)
    end
    Nm = map(x->min(x...), Adims)
    N1 = map((x, y) -> [x[1], y], Adims, Nm)
    N2 = map((x, y) -> [y, x[2]], Adims, Nm)
    Asize = A.size
    sm = min(Asize...)
    Utensor = vcat(map(vec, Utensor)...)
    Stensor = vcat(map(vec, Stensor)...)
    Vtensor = vcat(map(vec, Vtensor)...)
    U1Array(qn, A.dir, Utensor, (Asize[1], sm), N1, div, A.ifZ2), U1Array(qn, A.dir, Stensor, (sm, sm), [[Nm[i], Nm[i]] for i in 1:length(qn)], div, A.ifZ2), U1Array(qn, A.dir, Vtensor, (sm, Asize[2]), N2, div, A.ifZ2)
end


function order2(a) 
    return  a<5 ? a : (a-4)
end  
 

function update_row(Gamma, lamda, Udtau, position, sitetype, atype)
    # up: left=1, lm=1
    # down: left=0, lm=3
    if position =="up"
        left = 0
        right = 1
        lm = 1
    elseif position == "down"
        left = 1
        right = 0
        lm = 3
    else
        println("Wrong Position in update row!")
    end
     
    
    GA = Gamma[left]
    GB = Gamma[right]

    D = GA.size[4]
    d = U_2site.size[1]
    
    # d, D, d, D
    swap_dD = U1swapgate(atype, ComplexF64, d, D; 
                        indqn = [getqrange(sitetype, d)..., getqrange(sitetype, D)..., getqrange(sitetype, d)..., getqrange(sitetype, D)], 
                        indims = [getblockdims(sitetype, d)..., getblockdims(sitetype, D), getblockdims(sitetype, d)..., getblockdims(sitetype, D)],
                        ifZ2=sitetype.ifZ2
                        )
     
	 
    # step 1: Fig.29(a)
    GAA = ein"pludr,pdsy -> sluyr"(GA, swap_dD) 

    # step 2, Fig.34(a): obtain left part Theta  clockwise: dlurs   
    Gamma_0 = ein"sludr,dx,ly,uz -> xyzrs"(GAA, lamda[ order2(lm+1) ], lamda[ order2(lm+2) ], lamda[ order2(lm+3) ])

    # step 2': Fig.29(c) SVD/QR with no truncation 
    temp = reshape(Gamma_0, D^3, D*d)
    Q, R = qrpos(temp)
    WA = reshape(Q, D, D, D, D*d)
    MA = reshape(R, D*d, D, d)

    # step 3, Fig.34(a): right part Gamma clock-wise   slurd
    Gamma_1 = ein"pludr,ux,ry,dz -> plxyz"(GB, lamda[ order(lm+1) ], lamda[ order(lm+2) ], lamda[ order(lm+3) ])

    # step 3': Fig.29(d) SVD/RQ with no truncation 
    temp = reshape(Gamma_1, d*D, D^3)
    Rt, Qt = lqpos(temp)
    UB = reshape(Qt, D*d, D, D, D)
    NB = reshape(Rt, d, D, D*d)

    # step 4, Fig.34(a) or Towards Fig B.1(II)
    # MA =l, r,s    # s,l,r
    Theta_0 = ein"lrs,rx,pxy -> lspy"(MA, lamda[lm], NB) # anticlock-wise, downsides
    # (e,s,e,s)
    Theta = ein"lspr,tsqp -> ltqr"(Theta_0, Udtau) # anticlock-wise, downsides

    temp = reshape(Theta, d*D, d*D)







    
end