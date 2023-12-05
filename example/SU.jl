using ADFPEPS
using ADFPEPS:double_ipeps_energy,swapgate, generate_vertical_rules, generate_horizontal_rules, U1swapgate
using CUDA
using Random
using TeneT
using TeneT: randU1, zerosU1, IU1, qrpos, lqpos,svd!, initialA, zerosinitial, asU1Array, zerosinitial, randU1DiagMatrix, invDiagU1Matrix, sqrtDiagU1Matrix
using LinearAlgebra
using OMEinsum
using Printf 
  
 
function RandomInit(sitetype, atype, dtype, D, d)
    lambda = Dict()
    for i = 1:4    
         lambda[i] = IU1(sitetype, atype, dtype, D; dir = [-1,1]) 
    end

    Gamma = Dict()
    for i = 1:2   
        temp = randU1(sitetype, atype, dtype, d, D, D, D, D ; dir = [1, -1, -1, 1, 1])     
        Gamma[i] = temp/norm(temp)
    end 
    return Gamma, lambda
end 


function genGate(model, dtau, atype, dtype )
     
    #dtau = 0

    h = atype{dtype}.(hamiltonian(model))
 
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
    
    return  U_local, U_2sites_1
end

 
function order(a) 
    return  a<5 ? a : (a-4)
end  
 
function qndims(GA, ind)     
    indqn_D = Int64[]
    indims_D = Int64[]
    qn = GA.qn 
    dims = GA.dims 
    for i in 1:length(GA.qn)
        if qn[i][ind] in indqn_D
            continue
        end 
        push!(indqn_D, qn[i][ind]) 
        push!(indims_D, dims[i][ind]) 
    end   
    return indqn_D, indims_D
end

function update_row!(Gamma, lambda, Udtau, sitetype, atype, bond; position)
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

    #@show  norm(lambda[lm])
     
    
    GA = Gamma[left]
    GB = Gamma[right]

    D = GA.size[4]
    d = GA.size[1]  
     
     
    # @show indqn_D indims_D      
    
    # d, D, d, D
    # @show GA.qn GA.dims 
    indqn_D, indims_D = qndims(GA, 4)  
    swap_dD = U1swapgate(atype, ComplexF64, d, D; 
                        indqn = [getqrange(sitetype, d)..., indqn_D, getqrange(sitetype, d)..., indqn_D], 
                        indims = [getblockdims(sitetype, d)..., indims_D, getblockdims(sitetype, d)..., indims_D],
                        ifZ2=sitetype.ifZ2
                        ) 
    #@show swap_dD.size swap_dD.dir 
    # step 1: Fig.29(a)
    #@show swap_dD.qn swap_dD.dims 
    GAA = ein"pludr,pdsy -> sluyr"(GA, swap_dD) 

    # step 2, Fig.34(a): obtain left part Theta  clockwise: dlurs 
    Gamma_0 = ein"((sludr,dx),yl),zu -> xyzrs"(GAA, lambda[ order(lm+1) ], lambda[ order(lm+2) ], lambda[ order(lm+3) ])
 
    #@show Gamma_0.dir 
    # step 2': Fig.29(c) SVD/QR with no truncation 
    Gamma_0 = reshape(Gamma_0, D^3, D*d) # 
    #@show temp.size 
    Q, R = qrpos(Gamma_0)
    
    WA = reshape(Q, D, D, D, D*d) 
    MA = reshape(R, D*d, D, d)
    # @show WA.dir MA.dir 
    # temp = ein"dlur,rxs -> dluxs"(WA,MA)  
    # @show norm(temp - Gamma_0)

    # step 3, Fig.34(a): right part Gamma clock-wise   slurd
    Gamma_1 = ein"((pludr,xu),ry),dz -> plxyz"(GB, lambda[ order(lm+1) ], lambda[ order(lm+2) ], lambda[ order(lm+3) ])

    # step 3': Fig.29(d) SVD/RQ with no truncation      
    Q, R = qrpos(reshape(Gamma_1, d*D, D^3)')
    # @show Q.dir, R.dir 
    # @show Q'.dir R'.dir 
    UB = reshape(Q', D*d, D, D, D)
    NB = reshape(R', d, D, D*d)

    # step 4, Fig.34(a) or Towards Fig B.1(II)
    # MA =l, r,s    # s,l,r
    # @show MA.size lambda[lm].size NB.size
    # @show MA.dir lambda[lm].dir NB.dir
    Theta_0 = ein"(lrs,rx),pxy -> lspy"(MA, lambda[lm], NB) # anticlock-wise,  
  
    Theta = ein"lspr,psqt -> ltqr"(Theta_0, Udtau) # anticlock-wise,
     
    shape = Theta.size
    temp = reshape(Theta, shape[2]*shape[1], shape[3]*shape[4]) #  
    U, S, V = svd!(temp; trunc=bond, middledir=1)  #  
    #@show U.size S.size V'.size 
    # @show U.dir S.dir V'.dir
    #@show norm(U * Diagonal(S) * V' - temp)     
    MA_new = reshape(U, shape[1], shape[2], bond)
    NB_new = reshape(V', bond, shape[3], shape[4])
    #@show S 
    S = Diagonal(S)
    #@show S 
    # @show MA_new.size S.size NB_new.size 
    # @show MA_new.dir S.dir NB_new.dir       

    # step 5, Fig.(35)
    inv_lambda2 = invDiagU1Matrix( lambda[ order(lm+1) ] )
    inv_lambda3 = invDiagU1Matrix( lambda[ order(lm+2) ] )
    inv_lambda4 = invDiagU1Matrix( lambda[ order(lm+3) ] )

    # temp = ein"lr,rm -> lm"(inv_lambda2,lambda[ order(lm+1) ])  
    # @show temp 
 
    #  WA:(d,l,u,r)  # MA_new:(lpr)
    GAA_new = ein"(((dlur,dx),yl),zu),rpw -> pyzxw"(WA, inv_lambda2, inv_lambda3, inv_lambda4, MA_new)  
    # NB_new:(lpr)  # UB:(l,u,r,d )
    GB_new = ein"(((mpl,lurd),xu),ry),dz -> pmxzy"(NB_new, UB, inv_lambda2, inv_lambda3, inv_lambda4)
 
  
    # step 6, Fig.(32b)
    indqn_D, indims_D = qndims(GAA_new, 4)  
    swap_dD = U1swapgate(atype, ComplexF64, d, D; 
                        indqn = [getqrange(sitetype, d)..., indqn_D, getqrange(sitetype, d)..., indqn_D], 
                        indims = [getblockdims(sitetype, d)..., indims_D, getblockdims(sitetype, d)..., indims_D],
                        ifZ2=sitetype.ifZ2
                        ) 
    GA_new = ein"pludr,pdsy -> sluyr"(GAA_new, swap_dD) 

    # @show norm(GA_new - GA ) norm(GB_new - GB )
 
    Gamma[left] = GA_new
    Gamma[right] = GB_new
 
    #@show   norm(GA_new)*norm(S)*norm(GB_new)
    temp = norm(S)
    S = S/temp

    #@show  norm(S)

    lambda[lm] = S
      
    return  temp 
end


function update_column!(Gamma, lambda, Udtau, sitetype, atype, bond; position)
    if position =="left"
        left = 1
        right = 2
        lm = 2
    elseif position =="right"
        left = 2
        right = 1
        lm = 4
    else
        println("Wrong Position in update row!")
    end 

    GA = Gamma[left]
    GB = Gamma[right]

    D = GA.size[2]
    d = Udtau.size[1]
    
    # d, D, d, D
    indqn_D, indims_D = qndims(GB, 2)  
    swap_dD = U1swapgate(atype, ComplexF64, d, D; 
                        indqn = [getqrange(sitetype, d)..., indqn_D, getqrange(sitetype, d)..., indqn_D], 
                        indims = [getblockdims(sitetype, d)..., indims_D, getblockdims(sitetype, d)..., indims_D],
                        ifZ2=sitetype.ifZ2
                        )   	 
    # step 1: Fig.29(a)
    GBB = ein"pludr,pxsl -> sxudr"(GB, swap_dD) 

    # step 2, Fig.34(a): obtain Theta up  clockwise  [lurds]
    Gamma_0 = ein"((sludr,xl),yu),rz -> xyzds"(GA, lambda[ order(lm+1) ], lambda[ order(lm+2) ], lambda[ order(lm+3) ])
  
    # step 2': Fig.29(c) SVD/QR with no truncation 
    Gamma_0 = reshape(Gamma_0, D^3, D*d) # 
    #@show temp.size 
    Q, R = qrpos(Gamma_0)
    
    WA = reshape(Q, D, D, D, D*d) 
    MA = reshape(R, D*d, D, d)
    # @show WA.dir MA.dir 
    # temp = ein"dlur,rxs -> dluxs"(WA,MA)  

    # step 3, Fig.34(a): clock-wise  down: [surdl] 
    Gamma_1 = ein"((pludr,rx),dy),zl -> puxyz"(GBB, lambda[ order(lm+1) ], lambda[ order(lm+2) ], lambda[ order(lm+3) ])

    # step 3': Fig.29(d) SVD/RQ with no truncation      
    Q, R = qrpos(reshape(Gamma_1, d*D, D^3)')
    # @show Q.dir, R.dir 
    # @show Q'.dir R'.dir 
    UB = reshape(Q', D*d, D, D, D)
    NB = reshape(R', d, D, D*d)

    # step 4, Fig.34(a) or Towards Fig B.1(II)
    # MA: (u,d,p)   # NB: (s,u,d)
    # @show MA.size lambda[lm].size NB.size
    # @show MA.dir lambda[lm].dir NB.dir
    Theta_0 = ein"(udp,dy),syx -> upsx"(MA, lambda[lm], NB) # anticlock-wise, left 
    # !!!!!!!!!!!!!!!!!!!!!!!!
    Theta = ein"upsd,psxy -> uxyd"(Theta_0, Udtau) # anticlock-wise 
 
    shape = Theta.size
    temp = reshape(Theta, shape[2]*shape[1], shape[3]*shape[4]) #  
    U, S, V = svd!(temp; trunc=bond, middledir=1)  #  
    MA_new = reshape(U, shape[1], shape[2], bond)
    NB_new = reshape(V', bond, shape[3], shape[4])    
    S = Diagonal(S)
    

    # step 5, Fig.(35)
    inv_lambda2 = invDiagU1Matrix( lambda[ order(lm+1) ] )
    inv_lambda3 = invDiagU1Matrix( lambda[ order(lm+2) ] )
    inv_lambda4 = invDiagU1Matrix( lambda[ order(lm+3) ] )

    #  WA:(l,u,r,d  ),  l3, l4, l1
    GA_new = ein"(((lurd,xl),yu),rz),dpw -> pxywz"(WA, inv_lambda2, inv_lambda3, inv_lambda4, MA_new)  
    # UB:(u,r,d,l)
    GBB_new = ein"(((mpu,urdl),rx),dy),zl -> pzmyx"(NB_new, UB, inv_lambda2, inv_lambda3, inv_lambda4)

    # step 6, Fig.(32b)
    indqn_D, indims_D = qndims(GBB_new, 2)  
    swap_dD = U1swapgate(atype, ComplexF64, d, D; 
                        indqn = [getqrange(sitetype, d)..., indqn_D, getqrange(sitetype, d)..., indqn_D], 
                        indims = [getblockdims(sitetype, d)..., indims_D, getblockdims(sitetype, d)..., indims_D],
                        ifZ2=sitetype.ifZ2
                        )   
    GB_new = ein"pludr,pxsl -> sxudr"(GBB_new, swap_dD) 

    Gamma[left] = GA_new
    Gamma[right] = GB_new

    temp = norm(S)
    S = S/temp
 
    lambda[lm] = S
      
    return  temp 
end 




# --------------------- update once --------------------
function update_once_2nd!(Gamma, lambda, U_local, U_2sites_1, sitetype, atype, D, doEstimate)
    Gamma[1] = ein"pludr,ps -> sludr"(Gamma[1], U_local)
    Gamma[2] = ein"pludr,ps -> sludr"(Gamma[2], U_local) 

    if !doEstimate
        # step 5, may not needed
        Gamma[1] = Gamma[1]/norm(Gamma[1])
        Gamma[2] = Gamma[2]/norm(Gamma[2])
    end  
    
    temp = 1.0

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
    
    return temp
     
end


function back_to_state(Gamma, lambda)

    sqrt_lambda = Dict()
    for key in keys(lambda)
        sqrt_lambda[key] = sqrtDiagU1Matrix(lambda[key])
    end
    # order: ulpdr
    A = ein"(((pludr,xl),yu),dz),rw->yxpzw"(Gamma[1], sqrt_lambda[3], sqrt_lambda[4], sqrt_lambda[2], sqrt_lambda[1])
    A = A/norm(A)
    B = ein"(((pludr,xl),yu),dz),rw->yxpzw"(Gamma[2], sqrt_lambda[1], sqrt_lambda[2], sqrt_lambda[4], sqrt_lambda[3])
    B = B/norm(B)
    
    return A, B 
end
 


function SU_ABBA!(Gamma, lambda, model, SUparameter)

    sitetype = SUparameter["sitetype"]
    atype = SUparameter["atype"]
    dtype = SUparameter["dtype"]
    D = SUparameter["D"]

    dtau = SUparameter["dtau"]
    tratio = SUparameter["tratio"]
    Mindtau = SUparameter["Mindtau"]
    NoUp = SUparameter["NoUp"]
    doEstimate = SUparameter["doEstimate"]  
    tolerance_Es = SUparameter["tolerance_Es"] 
    count_upper = SUparameter["count_upper"]
    count_lower = SUparameter["count_lower"]
     

    E_E = []
    push!(E_E, 10.0)
    Entropy = []
    push!(Entropy, 10.0)
    count = 0    
 
    U_local, U_2sites_1 = genGate(model, dtau, atype, dtype)

    # a = IU1(sitetype, atype, dtype, 9; dir = [-1,1])
    # @show norm(U_local - a )
    # aa = ein"ab,cd ->acbd"(a, a) 
    # @show norm(aa - U_2sites_1)

    for i in range(1, NoUp)
        count += 1

        temp = update_once_2nd!(Gamma, lambda, U_local, U_2sites_1, sitetype, atype, D, doEstimate)  
        
        Estimator = -0.5/dtau * log( temp )
        # println("E_E = ", Estimator )
        push!(E_E, Estimator)
        dE_E = ( E_E[end] - E_E[end-1] ) / 1

        temp = 0.0
        for k = 1:4 
            for val in lambda[k].tensor 
                t1 = real(val)   
                if t1 > 1.e-16             
                    temp += -sum(t1 * log(t1))  
                end 
            end 
        end
        push!(Entropy, temp)       
        dEnt = Entropy[end] - Entropy[end-1]

        if i % 10 == 0
            @printf "\n-------> i = %d, Δτ =%.4e, E_E=%.8f, dE_E = %.8e "  i  dtau  E_E[end]  dE_E  
            @printf "\n                            Entropy=%.8f, dEnt = %.8e "  Entropy[end] dEnt   
        end 

        if count > count_lower
            if abs( dE_E ) < tolerance_Es ||  count > count_upper
                dtau = dtau * tratio 
                @printf "\n =====>> i = %d, Δτ =%.4e, count=%d, dE_E = %.8e "  i  dtau count  dE_E   
                count = 0
                print("\nReduced Δτ to  ", dtau )
                U_local, U_2sites_1 = genGate(model, dtau, atype, dtype)
            end
        end 

        if dtau <  Mindtau 
             break
        end 

    end    
    
    return Gamma, lambda
    
end


  
  
atype = Array 
dtype =  ComplexF64
D = 2
d = 9
χ = 20
sitetype = tJbilayerZ2()
model = tJ_bilayer(3.0,1.0,0.0,2.0,0.0)

# init Gamma, lambda 
Gamma, lambda = RandomInit(sitetype, atype, dtype, D, d)

# Simple update
SUparameter=Dict()
SUparameter["sitetype"] = sitetype
SUparameter["atype"] = atype
SUparameter["dtype"] = dtype 
SUparameter["D"] = D
SUparameter["dtau"] = 0.4
SUparameter["tratio"] = 0.7
SUparameter["Mindtau"] = 0.0001
SUparameter["NoUp"] = 200   
SUparameter["doEstimate"] = true
SUparameter["tolerance_Es"] = 1.0e-8
SUparameter["count_upper"] = 200
SUparameter["count_lower"] = 50
SU_ABBA!(Gamma, lambda, model, SUparameter) 


# ！！ measure  ！！
A, B = back_to_state(Gamma, lambda)
 
L = length(A.tensor)
ipeps = zeros(dtype, L, 2)  
ipeps[:,1] = A.tensor 
ipeps[:,2] = B.tensor  

symmetry = :U1
tol = 1e-10
maxiter = 50
miniter = 1
@show A.dims B.dims 
indD, dimsD = qndims(A, 1)  # can be wrong 
indχ = [0,1]
dimsχ = [10, 10]
folder = "../data/$sitetype/$(model)_$(D)/"
key = (folder, model, 2, 2, symmetry, sitetype, atype, d, D, χ, tol, maxiter, miniter, indD, indχ, dimsD, dimsχ)
consts = initial_consts(key)
 
E = double_ipeps_energy(ipeps, consts, key)	
println("E = \n", E)


println("\n886")     