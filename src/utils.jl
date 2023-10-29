import Base.show

show(x::AbstractArray) = show(IOContext(stdout, :compact => false), "text/plain", x)