# contains some utils for Fermionic Tensor Network Construction

using BitBasis
using CUDA
using TeneT
using TeneT: dtr, getparity
using Zygote

include("contractrules.jl")

_arraytype(::Array{T}) where {T} = Array
_arraytype(::CuArray{T}) where {T} = CuArray

"""
	T_parity_conserving(T::Array)

Transform ipeps into parity conserving form

"""
function T_parity_conserving(T::AbstractArray)
	s = size(T)
	p = zeros(s)
	bit = ceil(Int, log2(s[3]))
	for index in CartesianIndices(T)
		i = index.I .- 1
		sum((sum(bitarray(i[3],bit)), sum(i[[1,2,4,5]]))) % 2 == 0 && (p[index] = 1)
	end
	p = _arraytype(T)(p)

	return reshape(p.*T,s...)
end

function particle_conserving!(T::Union{Array,CuArray})
	bits = map(x -> ceil(Int, log2(x)), size(T))
    T[map(x->!(sum(sum.(bitarray.((Tuple(x).-1), bits))) in [0,1]), CartesianIndices(T))] .= 0
    # T[map(x->sum(sum.(bitarray.((Tuple(x).-1), bits))) !== 2, CartesianIndices(T))] .= 0
    return T
end
particle_conserving(T) = particle_conserving!(copy(T))

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
function swapgate(d::Int, D::Int)
	S = ein"ij,kl->ikjl"(Matrix{ComplexF64}(I,d,d),Matrix{ComplexF64}(I,D,D))
	for j = 1:D, i = 1:d
		# sum(bitarray(i-1,Int(ceil(log2(d)))))%2 != 0 && sum(bitarray(j-1,Int(ceil(log2(D)))))%2 != 0 && (S[i,j,:,:] .= -S[i,j,:,:])
        index_to_parity(i) != 0 && index_to_parity(j) != 0 && (S[i,j,i,j] = -1)
	end
	return S
end

function U1swapgate(atype, dtype, d::Int, D::Int; indqn::Vector{Vector{Int}}, indims::Vector{Vector{Int}})
    (d, D, d, D) != Tuple(map(sum, indims)) && throw(Base.error("U1swapgate indims is not valid"))
    dir = [-1, -1, 1, 1]
    qn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:4]
        if sum(qni .* dir) == 0 && qni[1] == qni[3] && qni[2] == qni[4]
            bulkdims = [indims[j][i.I[j]] for j in 1:4]
            push!(qn, qni)
            push!(dims, bulkdims)
            tensori = atype(ein"ij,kl->ikjl"(Matrix{dtype}(I, bulkdims[1], bulkdims[1]), Matrix{dtype}(I,  bulkdims[2], bulkdims[2])))
            isodd(qni[1]) && isodd(qni[2]) && (tensori .= -tensori)
            push!(tensor, tensori)
        end
    end
    p = sortperm(qn)
    tensor = vcat(map(vec, tensor[p])...)
    U1Array(qn[p], dir, tensor, (d, D, d, D), dims[p], 1)
end

function swapgatedD(d::Int, D::Int)
	S = ein"ij,kl->ikjl"(Matrix{ComplexF64}(I,d,d),Matrix{ComplexF64}(I,D,D))
	for j = 1:D, i = 1:d
		sum(bitarray(i-1,Int(ceil(log2(d)))))%2 != 0 && (j-1) % 2 != 0 && (S[i,j,:,:] .= -S[i,j,:,:])
	end
	return S
end

function swapgateDD(D::Int)
	S = ein"ij,kl->ikjl"(Matrix{ComplexF64}(I,D,D),Matrix{ComplexF64}(I,D,D))
	for j = 1:D, i = 1:D
		(i-1) % 2 != 0 && (j-1) % 2 != 0 && (S[i,j,:,:] .= -S[i,j,:,:])
	end
	return S
end

function Z2bitselectiond(maxN::Int)
    q = [sum(bitarray(i-1,ceil(Int,log2(maxN)))) % 2 for i = 1:maxN]
    return [(q .== 0),(q .== 1)]
end

function Z2bitselectionD(maxN::Int)
    q = [(i-1) % 2 for i = 1:maxN]
    return [(q .== 0),(q .== 1)]
end

function Z2t(A::Z2Array{T,N}) where {T,N}
    atype = _arraytype(A.tensor[1])
    tensor = zeros(T, size(A))
    parity = A.parity
    qlist = [i == 3 ? Z2bitselectiond(size(A)[i]) : Z2bitselectionD(size(A)[i]) for i = 1:N]
    for i in 1:length(parity)
        tensor[[qlist[j][parity[i][j]+1] for j = 1:N]...] = Array(A.tensor[i])
    end
    atype(tensor)
end

function t2Z(A::AbstractArray{T,N}) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [i == 3 ? Z2bitselectiond(size(A)[i]) : Z2bitselectionD(size(A)[i]) for i = 1:N]
    parity = getparity(N)
    tensor = [atype(Aarray[[qlist[j][parity[i][j]+1] for j = 1:N]...]) for i in 1:length(parity)]
    dims = map(x -> [size(x)...], tensor)
    Z2Array(parity, tensor, size(A), dims, 1)
end

function t2ZSdD(A::AbstractArray{T,N}) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [i in [1,3] ? Z2bitselectiond(size(A)[i]) : Z2bitselectionD(size(A)[i]) for i = 1:N]
    parity = getparity(N)
    tensor = [atype(Aarray[[qlist[j][parity[i][j]+1] for j = 1:N]...]) for i in 1:length(parity)]
    dims = map(x -> [size(x)...], tensor)
    Z2Array(parity, tensor, size(A), dims, 1)
end

function t2ZSDD(A::AbstractArray{T,N}) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [Z2bitselectionD(size(A)[i]) for i = 1:N]
    parity = getparity(N)
    tensor = [atype(Aarray[[qlist[j][parity[i][j]+1] for j = 1:N]...]) for i in 1:length(parity)]
    dims = map(x -> [size(x)...], tensor)
    Z2Array(parity, tensor, size(A), dims, 1)
end

"""
    function fdag(T::Array{V,5}) where V<:Number

Obtain dag tensor for local peps tensor in Fermionic Tensor Network(by inserting swapgates). The input tensor has indices which labeled by (lurdf)
legs are counting from f and clockwisely.

input legs order: ulfdr
output legs order: ulfdr
"""
function fdag(T::AbstractArray, SDD::AbstractArray)
	ein"(ulfdr,luij),pqrd->jifqp"(conj(T), SDD, SDD)
end

"""
    function bulk(T::Array{V,5}) where V<: Number
    
Obtain bulk tensor in peps, while the input tensor has indices which labeled by (lurdf).
This tensor is ready for GCTMRG (general CTMRG) algorithm
"""
function bulk(T::AbstractArray, SDD::AbstractArray, indD, dimsD)
	nu,nl,nf,nd,nr = size(T)
	Tdag = fdag(T, SDD)
    doublelayer = ein"((abcde,fgchi),lfbm), dkji-> glhjkema"(T,Tdag,SDD,SDD)
    indqn = [indD for _ in 1:8]
    indims = [dimsD for _ in 1:8]
    symmetryreshape(doublelayer, nl^2, nd^2, nr^2, nu^2; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))[1]
end

"""
    function bulkop(T::Array{V,5}) where V<: Number
    
Obtain bulk tensor in peps, while the input tensor has indices which labeled by (lurdf).
This tensor is ready for GCTMRG (general CTMRG) algorithm
"""
function bulkop(T::AbstractArray, SDD::AbstractArray, indD, dimsD)
	nu,nl,nf,nd,nr = size(T)
	Tdag = fdag(T, SDD)
    doublelayerop = ein"((abcde,fgnhi),lfbm), dkji-> glhjkemacn"(T,Tdag,SDD,SDD)
    indqn = [[indD for _ in 1:8]; getqrange(nf, nf)]
    indims = [[dimsD for _ in 1:8]; getblockdims(nf, nf)]
    symmetryreshape(doublelayerop, nl^2,nd^2,nr^2,nu^2,nf,nf; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))[1]
	# return	_arraytype(T)(reshape(ein"((abcde,fgnhi),bflm),dijk -> glhjkencma"(T,Tdag,SDD,SDD),nu^2,nl^2,nd^2,nr^2,nf,nf))
end

function index_to_parity(n::Int)
    n -= 1
    n == 0 && return 0

    ternary = []
    while n > 0
        remainder = n % 3
        remainder == 2 && (remainder = 1)
        pushfirst!(ternary, remainder)
        n = div(n, 3)
    end

    return sum(ternary) % 2
end