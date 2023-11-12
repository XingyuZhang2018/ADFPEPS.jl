# contains some utils for Fermionic Tensor Network Construction

using BitBasis
using CUDA
using TeneT
using TeneT: dtr
using Zygote

"""
    function swapgate(n1::Int,n2::Int)

Generate a tensor which represent swapgate in Fermionic Tensor Network. n1,n2 should be power of 2.
The generated tensor have 4 indices. (ijkl).
S(ijkl) = delta(ik)*delta(jl)*parity(gate)

"""
function swapgate(siteinds, atype, dtype, d::Int, D::Int)
	S = ein"ij,kl->ikjl"(Matrix{dtype}(I,d,d),Matrix{dtype}(I,D,D))
	for j = 1:D, i = 1:d
        abs(indextoqn(siteinds, i)) % 2 != 0 && abs(indextoqn(siteinds, j)) % 2 != 0 && (S[i,j,i,j] = -1)
	end
	return atype(S)
end

function U1swapgate(atype, dtype, d::Int, D::Int; indqn::Vector{Vector{Int}}, indims::Vector{Vector{Int}}, ifZ2::Bool)
    (d, D, d, D) != Tuple(map(sum, indims)) && throw(Base.error("U1swapgate indims is not valid"))
    dir = [-1, -1, 1, 1]
    qn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:4]
        qnsum = ifZ2 ? sum(qni) % 2 : sum(qni .* dir)
        if qnsum == 0 && qni[1] == qni[3] && qni[2] == qni[4]
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
    U1Array(qn[p], dir, tensor, (d, D, d, D), dims[p], 1, ifZ2)
end

function Z2swapgate(atype, dtype, d::Int, D::Int; indims::Vector{Vector{Int}})
    (d, D, d, D) != Tuple(map(sum, indims)) && throw(Base.error("Z2swapgate indims is not valid"))
    parity = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(2 for _ in 1:4))
        parityi = [[0,1][i.I[j]] for j in 1:4]
        if sum(parityi) % 2 == 0 
            bulkdims = [indims[j][i.I[j]] for j in 1:4]
            push!(parity, parityi)
            push!(dims, bulkdims)
            if parityi[1] == parityi[3] && parityi[2] == parityi[4]
                tensori = atype(ein"ij,kl->ikjl"(Matrix{dtype}(I, bulkdims[1], bulkdims[1]), Matrix{dtype}(I,  bulkdims[2], bulkdims[2])))
                isodd(parityi[1]) && isodd(parityi[2]) && (tensori .= -tensori)
            else
                tensori = atype(zeros(dtype, bulkdims[1], bulkdims[2], bulkdims[3], bulkdims[4]))
            end
            push!(tensor, tensori)
        end
    end
    p = sortperm(parity)
    Z2Array(parity[p], tensor[p], (d, D, d, D), dims[p], 1)
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
function bulkop(T::AbstractArray, SDD::AbstractArray, indD, dimsD, sitetype)
	nu,nl,nf,nd,nr = size(T)
	Tdag = fdag(T, SDD)
    doublelayerop = ein"((abcde,fgnhi),lfbm), dkji-> glhjkemacn"(T,Tdag,SDD,SDD)
    indqn = [[indD for _ in 1:8]; getqrange(sitetype, nf, nf)]
    indims = [[dimsD for _ in 1:8]; getblockdims(sitetype, nf, nf)]
    symmetryreshape(doublelayerop, nl^2,nd^2,nr^2,nu^2,nf,nf; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))[1]
	# return	_arraytype(T)(reshape(ein"((abcde,fgnhi),bflm),dijk -> glhjkencma"(T,Tdag,SDD,SDD),nu^2,nl^2,nd^2,nr^2,nf,nf))
end
