# contains some utils for Fermionic Tensor Network Construction

using BitBasis
using CUDA
using VUMPS
using VUMPS: dtr, getparity
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

function Z2t(A::Z2tensor{T,N}) where {T,N}
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
    Z2tensor(parity, tensor, size(A), dims, 1)
end

function t2ZSdD(A::AbstractArray{T,N}) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [i in [1,3] ? Z2bitselectiond(size(A)[i]) : Z2bitselectionD(size(A)[i]) for i = 1:N]
    parity = getparity(N)
    tensor = [atype(Aarray[[qlist[j][parity[i][j]+1] for j = 1:N]...]) for i in 1:length(parity)]
    dims = map(x -> [size(x)...], tensor)
    Z2tensor(parity, tensor, size(A), dims, 1)
end

function t2ZSDD(A::AbstractArray{T,N}) where {T,N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [Z2bitselectionD(size(A)[i]) for i = 1:N]
    parity = getparity(N)
    tensor = [atype(Aarray[[qlist[j][parity[i][j]+1] for j = 1:N]...]) for i in 1:length(parity)]
    dims = map(x -> [size(x)...], tensor)
    Z2tensor(parity, tensor, size(A), dims, 1)
end

"""
    function fdag(T::Array{V,5}) where V<:Number

Obtain dag tensor for local peps tensor in Fermionic Tensor Network(by inserting swapgates). The input tensor has indices which labeled by (lurdf)
legs are counting from f and clockwisely.

input legs order: ulfdr
output legs order: ulfdr
"""
function fdag(T::Union{Array{V,5},CuArray{V,5}}, SDD::Union{Array{V,4},CuArray{V,4}}) where V<:Number
	ein"(ulfdr,luij),rdpq->jifqp"(conj(T), SDD, SDD)
end

function fdag(T::AbstractZ2Array, SDD::AbstractZ2Array)
	ein"(ulfdr,luij),rdpq->jifqp"(conj(T), SDD, SDD)
end

"""
    function bulk(T::Array{V,5}) where V<: Number
    
Obtain bulk tensor in peps, while the input tensor has indices which labeled by (lurdf).
This tensor is ready for GCTMRG (general CTMRG) algorithm
"""
function bulk(T::Union{Array{V,5},CuArray{V,5}}, SDD::Union{Array{V,4},CuArray{V,4}}) where V<:Number
	nu,nl,nf,nd,nr = size(T)
	Tdag = fdag(T, SDD)
	return	_arraytype(T)(reshape(ein"((abcde,fgchi),bflm),dijk -> glhjkema"(T,Tdag,SDD,SDD),nu^2,nl^2,nd^2,nr^2))
end

function bulk(T::AbstractZ2Array, SDD::AbstractZ2Array)
	nu,nl,nf,nd,nr = size(T)
	Tdag = fdag(T, SDD)
	return Z2reshape(ein"((abcde,fgchi),bflm),dijk -> glhjkema"(T,Tdag,SDD,SDD),nu^2,nl^2,nd^2,nr^2)
end

"""
    function bulkop(T::Array{V,5}) where V<: Number
    
Obtain bulk tensor in peps, while the input tensor has indices which labeled by (lurdf).
This tensor is ready for GCTMRG (general CTMRG) algorithm
"""
function bulkop(T::Union{Array{V,5},CuArray{V,5}}, SDD::Union{Array{V,4},CuArray{V,4}}) where V<:Number
	nu,nl,nf,nd,nr = size(T)
	Tdag = fdag(T, SDD)
	return	_arraytype(T)(reshape(ein"((abcde,fgnhi),bflm),dijk -> glhjkencma"(T,Tdag,SDD,SDD),nu^2,nl^2,nd^2,nr^2,nf,nf))
end