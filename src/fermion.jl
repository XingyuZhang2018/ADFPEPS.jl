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
	@assert prod(size(T))%2 == 0
	T = reshape(T,[2 for i = 1:Int(log2(prod(s)))]...)
	p = zeros(size(T))
	for index in CartesianIndices(T)
		if mod(sum([i for i in Tuple(index)].-1),2) == 0
			p[index] = 1
		end
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
	for i = 1:n1
		for j = 1:n2
			if sum(bitarray(i-1,Int(ceil(log(n1)/log(2)))))%2 !=0 && sum(bitarray(j-1,Int(ceil(log(n2)/log(2)))))%2 !=0
				S[i,j,:,:] .= -S[i,j,:,:]
			end
		end
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
	
	Tdag = ein"ulfdr,luij,rdpq->jifqp"(Tdag,_arraytype(T)(swapgate(nl,nu)),_arraytype(T)(swapgate(nr,nd)))
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
	eincode = EinCode(((1,2,3,4,5),(6,7,3,8,9),(2,6,10,11),(4,9,12,13)),(11,1,7,10,8,12,13,5))
	S1 = _arraytype(T)(swapgate(nl,nu))
	S2 = _arraytype(T)(swapgate(nd,nr))
	return	_arraytype(T)(reshape(einsum(eincode,(T,Tdag,S1,S2)),nu^2,nl^2,nd^2,nr^2))
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
	b = reshape([permutedims(bulk(T),(2,3,4,1))],1,1)
	b /= norm(b)
	# b += [1e-1*rand(ComplexF64, size(b[1]))]
	Mu, ALu, Cu, ARu, ALd, Cd, ARd, FLo, FRo, FL, FR = obs_env(b; χ=χ, maxiter=maxiter, verbose=true, savefile= true, infolder=infolder*"/$(model)/", outfolder=outfolder*"/$(model)/")

	E1 = permutedims(FLo[1,1],(3,2,1))
	E2 = ALu[1,1]
	E3 = ein"ij,jkl->ikl"(Cu[1,1],ARu[1,1])
	E4 = FRo[1,1]
	E5 = permutedims(ARd[1,1],(3,2,1))
	E6 = ein"ijk,kp->pji"(ALd[1,1],Cd[1,1])
	E7 = permutedims(FL[1,1],(3,2,1))
	E8 = FR[1,1]

	(E1,E2,E3,E4,E5,E6,E7,E8) = map(_arraytype(T),(E1,E2,E3,E4,E5,E6,E7,E8))
	return (E1,E2,E3,E4,E5,E6,E7,E8)
end

function double_ipeps_energy(T::Union{Array,CuArray}, model::HamiltonianModel;χ=80,maxiter=20,show_every=Inf,infolder=nothing,outfolder=nothing)
	T = parity_conserving(T)
    
	# @timeit timer "Obtain Enviroment" begin
		enviroments = ipeps_enviroment(T,model,χ=χ,maxiter=maxiter,show_every=5;infolder=infolder,outfolder=outfolder)
	# end
	
	# @timeit timer "Horizontal Contraction" begin
        h = reshape(_arraytype(T)(hamiltonian(model)), 4, 4, 4, 4)
		ρ = square_ipeps_contraction_horizontal(T,map(_arraytype(T),enviroments))
        E = ein"ijkl,ijkl -> "(ρ,h)[]
		n = ein"ijij -> "(ρ)[]
		e1 = E/n
	# end

	# @timeit timer "Vertical Contraction" begin
        h = reshape(_arraytype(T)(hamiltonian(model)), 4, 4, 4, 4)
		ρ = square_ipeps_contraction_vertical(T,map(_arraytype(T),enviroments))
        E = ein"ijkl,ijkl -> "(ρ,h)[]
		n = ein"ijij -> "(ρ)[]
		e2 = E/n
	# end

	println("VH=$(e1) \nVE=$(e2)")
	return real(e1+e2)
end

function square_ipeps_contraction_vertical(T,env)
	nu,nl,nf,nd,nr = size(T)
	χ = size(env[1])[1]
	(E1,E2,E3,E4,E5,E6,E7,E8) = map(x->reshape(x,χ,nl,nl,χ),env)

	optcode(x) = VERTICAL_RULES(map(_arraytype(T),x)...)
	result = optcode([T,fdag(T),swapgate(nl,nu),swapgate(nf,nu),
	swapgate(nf,nr),swapgate(nl,nu),T,fdag(T),swapgate(nl,nu),
	swapgate(nf,nr),swapgate(nf,nr),swapgate(nl,nu),E3,E8,E4,E6,E1,E7])
	return result
end

function square_ipeps_contraction_horizontal(T,env)
	nu,nl,nf,nd,nr = size(T)
	χ = size(env[1])[1]
	(E1,E2,E3,E4,E5,E6,E7,E8) = map(x->reshape(x,χ,nl,nl,χ),env)

	optcode(x) = HORIZONTAL_RULES(map(_arraytype(T),x)...)
	result = optcode([T,swapgate(nf,nu),
	fdag(T),swapgate(nf,nu),swapgate(nl,nu),
	swapgate(nl,nu),
	fdag(T),swapgate(nf,nu),
	swapgate(nf,nu),swapgate(nl,nu),T,
	swapgate(nl,nu),
	E1,E2,E3,E4,E5,E6])
	return result
end