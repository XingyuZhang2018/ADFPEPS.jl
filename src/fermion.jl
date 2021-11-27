# contains some utils for Fermionic Tensor Network Construction
using CUDA
using VUMPS
using BitBasis

include("contractrules.jl")

_arraytype(x::Array{T}) where {T} = Array
_arraytype(x::CuArray{T}) where {T} = CuArray

"""
    parity_conserving(T::Array)

Transform an arbitray tensor which has arbitray legs and each leg have index 1 or 2 into parity conserving form

# example

```julia
julia> T = rand(2,2,2)
2×2×2 Array{Float64, 3}:
[:, :, 1] =
 0.863822  0.133604
 0.865495  0.371586

[:, :, 2] =
 0.581621  0.819325
 0.197463  0.801167

julia> parity_conserving(T)
2×2×2 Array{Float64, 3}:
[:, :, 1] =
 0.863822  0.0
 0.0       0.371586

[:, :, 2] =
 0.0       0.819325
 0.197463  0.0
```
"""
function parity_conserving(T::Union{Array,CuArray}) where V<:Real
	s = size(T)
	p = zeros(s)
	bits = map(x -> Int(ceil(log2(x))), s)
	for index in CartesianIndices(T)
		i = Tuple(index) .- 1
		sum(sum.(bitarray.(i,bits))) % 2 == 0 && (p[index] = 1)
	end
	p = _arraytype(T)(p)

	return reshape(p.*T,s...)
end

"""
    function swapgate(n1::Int,n2::Int)

Generate a tensor which represent swapgate in Fermionic Tensor Network. n1,n2 should be power of 2.
The generated tensor have 4 indices. (ijkl).
S(ijkl) = delta(ik)*delta(jl)*parity(gate)

# example
```
julia> swapgate(2,4)
2×4×2×4 Array{Int64, 4}:
[:, :, 1, 1] =
 1  0  0  0
 0  0  0  0

[:, :, 2, 1] =
 0  0  0  0
 1  0  0  0

[:, :, 1, 2] =
 0  1  0  0
 0  0  0  0

[:, :, 2, 2] =
 0   0  0  0
 0  -1  0  0

[:, :, 1, 3] =
 0  0  1  0
 0  0  0  0

[:, :, 2, 3] =
 0  0   0  0
 0  0  -1  0

[:, :, 1, 4] =
 0  0  0  1
 0  0  0  0

[:, :, 2, 4] =
 0  0  0  0
 0  0  0  1
```
"""
function swapgate(n1::Int,n2::Int)
	S = ein"ij,kl->ikjl"(Matrix{Float64}(I,n1,n1),Matrix{Float64}(I,n2,n2))
	for j = 1:n2, i = 1:n1
		sum(bitarray(i-1,Int(ceil(log2(n1)))))%2 != 0 && sum(bitarray(j-1,Int(ceil(log2(n2)))))%2 != 0 && (S[i,j,:,:] .= -S[i,j,:,:])
	end
	return S
end


"""
    function fdag(T::Array{V,5}) where V<:Number

Obtain dag tensor for local peps tensor in Fermionic Tensor Network(by inserting swapgates). The input tensor has indices which labeled by (lurdf)
legs are counting from f and clockwisely.

input legs order: ulfdr
output legs order: ulfdr
"""
function fdag(T::Union{Array{V,5},CuArray{V,5}}) where V<:Number
	nu,nl,nf,nd,nr = size(T)
	Tdag = conj(T)
	
	S = _arraytype(T){ComplexF64}(swapgate(nl,nu))
	Tdag = ein"(ulfdr,luij),rdpq->jifqp"(Tdag, S, S)
	return Tdag	
end

"""
    function bulk(T::Array{V,5}) where V<: Number
    
Obtain bulk tensor in peps, while the input tensor has indices which labeled by (lurdf).
This tensor is ready for GCTMRG (general CTMRG) algorithm
"""
function bulk(T::Union{Array{V,5},CuArray{V,5}}) where V<:Number
	nu,nl,nf,nd,nr = size(T)
	Tdag = fdag(T)
	# u l s d r
	# eincode = EinCode(((1,2,3,4,5),(6,7,3,8,9),(2,6,10,11),(4,9,12,13)),(1,11,10,7,8,12,13,5))
	eincode = EinCode(((1,2,3,4,5),(6,7,3,8,9),(2,6,12,13),(4,9,10,11)),(13,1,7,12,8,10,11,5))
	S = _arraytype(T){ComplexF64}(swapgate(nl,nu))
	return	_arraytype(T)(reshape(einsum(eincode,(T,Tdag,S,S)),nu^2,nl^2,nd^2,nr^2))
end

"""
	calculate enviroment (E1...E6)
	a ────┬──── c 
	│     b     │ 
	├─ d ─┼─ e ─┤ 
	│     g     │ 
	f ────┴──── h 
	order: fda,abc,dgeb,hgf,ceh
"""
function ipeps_enviroment(T::AbstractArray, model;χ=20,maxiter=20,show_every=Inf,infolder=nothing,outfolder=nothing)
	Ni, Nj = size(T)
	b = reshape([permutedims(bulk(T[i]),(2,3,4,1)) for i = 1:Ni*Nj], (Ni, Nj))
	# VUMPS
	_, ALu, Cu, ARu, ALd, Cd, ARd, FLo, FRo, FL, FR = obs_env(b; χ=χ, maxiter=maxiter, miniter=1, verbose=true, savefile= true, infolder=infolder*"/$(model)_$(Ni)x$(Nj)/", outfolder=outfolder*"/$(model)_$(Ni)x$(Nj)/", downfromup = false, show_every=Inf)

	E1 = FLo
	E2 = reshape([ein"abc,cd->abd"(ALu[i],Cu[i]) for i = 1:Ni*Nj], (Ni, Nj))
	E3 = ARu
	E4 = FRo
	E5 = ARd
	E6 = reshape([ein"abc,cd->abd"(ALd[i],Cd[i]) for i = 1:Ni*Nj], (Ni, Nj))
	E7 = FL
	E8 = FR

	return (E1,E2,E3,E4,E5,E6,E7,E8)
end

function double_ipeps_energy(ipeps::AbstractArray, model::HamiltonianModel; Ni=1,Nj=1,χ=80,maxiter=20,show_every=Inf,infolder=nothing,outfolder=nothing)	
	T = reshape([parity_conserving(ipeps[:,:,:,:,:,i]) for i = 1:Ni*Nj], (Ni, Nj))
	E1,E2,E3,E4,E5,E6,E7,E8 = ipeps_enviroment(T,model,χ=χ,maxiter=maxiter,show_every=show_every;infolder=infolder,outfolder=outfolder)
	etol = 0
	
	atype = _arraytype(E1[1,1]){ComplexF64}
	hx = reshape(atype(-abs.(hamiltonian(model))), 4, 4, 4, 4)
	hy = reshape(atype(-abs.(hamiltonian(model))), 4, 4, 4, 4)
	# occ = reshape(atype(hamiltonian(Occupation())), 4, 4, 4, 4)
	for j = 1:Nj, i = 1:Ni
		ir = Ni + 1 - i
		jr = j + 1 - (j==Nj) * Nj
		
		ex = (E1[i,j],E2[i,j],E3[i,jr],E4[i,jr],E5[ir,jr],E6[ir,j],E7[i,j],E8[i,j])
		ρx = square_ipeps_contraction_horizontal(T[i,j],T[i,jr],ex)
		# ρ1 = reshape(ρ,16,16)
		# @show norm(ρ1-ρ1')
        Ex = ein"ijkl,ijkl -> "(ρx,hx)
		# Occ = ein"ijkl,ijkl -> "(ρx,occ)
		nx = ein"ijij -> "(ρx)
		etol += Array(Ex)[]/Array(nx)[]
		println("─ = $(Array(Ex)[]/Array(nx)[])") 
		# println("N = $(Array(Occ)[]/Array(nx)[])")

        ey = (E1[ir,j],E2[i,j],E3[i,j],E4[ir,j],E5[ir,jr],E6[i,j],E7[i,j],E8[i,j])
		ρy = square_ipeps_contraction_vertical(T[i,j],T[ir,j],ey)
		# ρ1 = reshape(ρ,16,16)
		# @show norm(ρ1-ρ1')
        Ey = ein"ijkl,ijkl -> "(ρy,hy)
		# Occ = ein"ijkl,ijkl -> "(ρy,occ)
		ny = ein"ijij -> "(ρy)
		etol += Array(Ey)[]/Array(ny)[]
		println("│ = $(Array(Ey)[]/Array(ny)[])")
		# println("N = $(Array(Occ)[]/Array(ny)[])")
	end

	return real(etol)/Ni/Nj
end

function square_ipeps_contraction_vertical(T1,T2,env)
	nu,nl,nf,nd,nr = size(T1)
	χ = size(env[1])[1]
	(E1,E2,E3,E4,E5,E6,E7,E8) = map(x->reshape(x,χ,nl,nl,χ),env)

	optcode(x) = VERTICAL_RULES(map(_arraytype(T1){ComplexF64},x)...)
	result = optcode([T1,fdag(T1),swapgate(nl,nu),swapgate(nf,nu),
	swapgate(nf,nr),swapgate(nl,nu),T2,fdag(T2),swapgate(nl,nu),
	swapgate(nf,nr),swapgate(nf,nr),swapgate(nl,nu),E2,E8,E4,E6,E1,E7])
	return result
end

function square_ipeps_contraction_horizontal(T1,T2,env)
	nu,nl,nf,nd,nr = size(T1)
	χ = size(env[1])[1]
	(E1,E2,E3,E4,E5,E6,E7,E8) = map(x->reshape(x,χ,nl,nl,χ),env)

	optcode(x) = HORIZONTAL_RULES(map(_arraytype(T1){ComplexF64},x)...)
	result = optcode([T1,swapgate(nf,nu),
	fdag(T1),swapgate(nf,nu),swapgate(nl,nu),
	swapgate(nl,nu),
	fdag(T2),swapgate(nf,nu),
	swapgate(nf,nu),swapgate(nl,nu),T2,
	swapgate(nl,nu),
	E1,E2,E3,E4,E5,E6])
	return result
end