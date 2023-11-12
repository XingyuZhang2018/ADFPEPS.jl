import Base.show
using CUDA

show(x::AbstractArray) = show(IOContext(stdout, :compact => false), "text/plain", x)

_arraytype(::Array{T}) where {T} = Array
_arraytype(::CuArray{T}) where {T} = CuArray
