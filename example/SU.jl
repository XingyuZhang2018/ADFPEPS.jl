using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules, U1swapgate
using CUDA
using Random
using TeneT
using TeneT: randU1, zerosU1, IU1, qrpos, lqpos,svd!, initialA, zerosinitial, asU1Array, zerosinitial, randU1DiagMatrix, invDiagU1Matrix
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
    temp = randU1(sitetype, atype, dtype, d, D, D, D, D ; dir = [1, -1, -1, 1, 1])
    push!(Gamma, temp)
end 

h = atype{ComplexF64}.(hamiltonian(model))


# ================   SU  =================
dtau = 0.2
nsteps = 100
 

# ------------------------  gate -------------------
U_local = exp(-0.5*dtau*h[2]  ) 
function gen_U_2sites( h2, dtau )
    d = size(h2)[1]
    temp = reshape(h2, d^2, d^2)
    temp = exp( -1.0 * dtau * temp ) 
    temp = reshape(temp, d, d, d, d)
    return temp
end
U_2sites_1 = gen_U_2sites(h[1], 0.5*dtau ) 
 
U_2sites_1 = asU1Array(sitetype, U_2sites_1; dir = [-1,-1,1,1]) 
U_local = asU1Array(sitetype, U_local; dir = [-1,1])

 
 
function order(a) 
    return  a<5 ? a : (a-4)
end  
 

function update_row!(Gamma, lamda, Udtau, sitetype, atype, bond; position)
    # up: left=1, lm=1
    # down: left=0, lm=3
    if position =="up"
        left = 1
        right = 2
        lm = 1
    elseif position == "down"
        left = 2
        right = 1
        lm = 3
    else
        println("Wrong Position in update row!")
    end
     
    
    GA = Gamma[left]
    GB = Gamma[right]

    D = GA.size[4]
    d = Udtau.size[1]
    
    # d, D, d, D
    swap_dD = U1swapgate(atype, ComplexF64, d, D; 
                        indqn = [getqrange(sitetype, d)..., getqrange(sitetype, D)..., getqrange(sitetype, d)..., getqrange(sitetype, D)...], 
                        indims = [getblockdims(sitetype, d)..., getblockdims(sitetype, D)..., getblockdims(sitetype, d)..., getblockdims(sitetype, D)...],
                        ifZ2=sitetype.ifZ2
                        )
     
	 
    # step 1: Fig.29(a)
    GAA = ein"pludr,pdsy -> sluyr"(GA, swap_dD) 

    # step 2, Fig.34(a): obtain left part Theta  clockwise: dlurs 
    Gamma_0 = ein"((sludr,dx),yl),zu -> xyzrs"(GAA, lamda[ order(lm+1) ], lamda[ order(lm+2) ], lamda[ order(lm+3) ])

 
    #@show Gamma_0.dir 
    # step 2': Fig.29(c) SVD/QR with no truncation 
    Gamma_0 = reshape(Gamma_0, D^3, D*d) # 
    #@show temp.size 
    Q, R = qrpos(Gamma_0)
    
    WA = reshape(Q, D, D, D, D*d) 
    MA = reshape(R, D*d, D, d)
    # @show WA.dir MA.dir 
    # temp = ein"dlur,rxs -> dluxs"(WA,MA)  

    # step 3, Fig.34(a): right part Gamma clock-wise   slurd
    Gamma_1 = ein"((pludr,xu),ry),dz -> plxyz"(GB, lamda[ order(lm+1) ], lamda[ order(lm+2) ], lamda[ order(lm+3) ])

    # step 3': Fig.29(d) SVD/RQ with no truncation      
    Q, R = qrpos(reshape(Gamma_1, d*D, D^3)')
    # @show Q.dir, R.dir 
    # @show Q'.dir R'.dir 
    UB = reshape(Q', D*d, D, D, D)
    NB = reshape(R', d, D, D*d)

    # step 4, Fig.34(a) or Towards Fig B.1(II)
    # MA =l, r,s    # s,l,r
    # @show MA.size lamda[lm].size NB.size
    # @show MA.dir lamda[lm].dir NB.dir
    Theta_0 = ein"(lrs,rx),pxy -> lspy"(MA, lamda[lm], NB) # anticlock-wise, downsides
    # !!!!!!!!!!!!!!!!!!!!!!!!
    Theta = ein"lspr,psqt -> ltqr"(Theta_0, Udtau) # anticlock-wise, downsides 
 
    shape = Theta.size
    temp = reshape(Theta, shape[2]*shape[1], shape[3]*shape[4]) #  
    U, S, V = svd!(temp; trunc=bond, middledir=1)  #  
    MA_new = reshape(U, shape[1], shape[2], bond)
    NB_new = reshape(V', bond, shape[3], shape[4])

    # step 5, Fig.(35)
    inv_lamda2 = invDiagU1Matrix( lamda[ order(lm+1) ] )
    inv_lamda3 = invDiagU1Matrix( lamda[ order(lm+2) ] )
    inv_lamda4 = invDiagU1Matrix( lamda[ order(lm+3) ] )

    #  WA:(d,l,u,r)  # MA_new:(lpr)
    GAA_new = ein"(((dlur,dx),yl),zu),rpw -> pyzxw"(WA, inv_lamda2, inv_lamda3, inv_lamda4, MA_new)  
    # NB_new:(lpr)  # UB:(l,u,r,d )
    GB_new = ein"(((mpl,lurd),xu),ry),dz -> pmxzy"(NB_new, UB, inv_lamda2, inv_lamda3, inv_lamda4)

    # step 6, Fig.(32b)
    GA_new = ein"pludr,pdsy -> sluyr"(GAA_new, swap_dD) 

    Gamma[left] = GA_new
    Gamma[right] = GB_new

    temp = norm(Diagonal(S))
    S = S/temp

    lamda[lm] = S
      
    return  temp 
end


function update_column!(Gamma, lamda, Udtau, sitetype, atype, bond; position)


    temp = 1

    return  temp
end 




# --------------------- update once --------------------
function update_once_2nd(Gamma, lambda, U_local, U_2sites_1, sitetype, atype, D, doEstimate)
    Gamma[1] = ein"pludr,ps -> sludr"(Gamma[1], U_local)
    Gamma[2] = ein"pludr,ps -> sludr"(Gamma[2], U_local) 

    if !doEstimate
        # step 5, may not needed
        Gamma[1] = Gamma[1]/norm(Gamma[1])
        Gamma[2] = Gamma[2]/norm(Gamma[2])
    end  
    
    temp = 1.0
    # down
    temp *= update_row!(Gamma, lambda, U_2sites_1, sitetype, atype, D, position="up")
    temp *= update_column!(Gamma, lambda, U_2sites_1, sitetype, atype, D, position="right") 
    temp *= update_row!(Gamma, lambda, U_2sites_1, sitetype, atype, D, position="down")    
    
    temp *= update_column!(Gamma, lambda, U_2sites_1, sitetype, atype, D, position="left")
    temp *= update_column!(Gamma, lambda, U_2sites_1, sitetype, atype, D, position="left")
    
    temp *= update_row!(Gamma, lambda, U_2sites_1, sitetype, atype, D, position="down")
    temp *= update_column!(Gamma, lambda, U_2sites_1, sitetype, atype, D, position="right")
    temp *= update_row!(Gamma, lambda, U_2sites_1, sitetype, atype, D, position="up")    
    
    
    Gamma[1] = ein"pludr,ps -> sludr"(Gamma[1], U_local)
    Gamma[2] = ein"pludr,ps -> sludr"(Gamma[2], U_local)
    
    if !doEstimate
        # step 5, may not needed
        Gamma[1] = Gamma[1]/norm(Gamma[1])
        Gamma[2] = Gamma[2]/norm(Gamma[2])
    end
    
    return Gamma, lambda, temp
     
end
 



Gamma, lamda, temp = update_once_2nd(Gamma, lambda, U_local, U_2sites_1, sitetype, atype, D, true)   

 
a = 1   