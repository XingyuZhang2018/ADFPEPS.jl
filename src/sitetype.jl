import TeneT: AbstractSiteType, indextoqn

"""
    indextoqn(::tJbilayerZ2(), i::Int)

index to quantum number for tJ Z2 sysmmetry 

state    qn  | remainder 
|00>  ->  0   | 0
|0↑>  ->  1   | 1
|0↓>  ->  1   | 2   
|↑0>  ->  1   | 3
|↑↑>  ->  0   | 4
|↑↓>  ->  0   | 5   
|↓0>  ->  1   | 6   
|↓↑>  ->  0   | 7   
|↓↓>  ->  0   | 8   
"""
struct tJbilayerZ2 <: AbstractSiteType 
    ifZ2::Bool
end
tJbilayerZ2() = tJbilayerZ2(true)
export tJbilayerZ2
function indextoqn(::tJbilayerZ2, i::Int)
    i -= 1
    i == 0 && return 0

    qni = []
    while i > 0
        remainder = i % 9
        if remainder in [1,2,3,6] 
            remainder = 1
        else
            remainder = 0
        end
        pushfirst!(qni, remainder)
        i = div(i, 9)
    end

    return sum(qni) % 2
end

"""
    indextoqn(::tJbilayerSz(), i::Int)

index to quantum number for tJ Sz sysmmetry 

state    qn  | remainder 
|00>  ->  0   | 0
|0↑>  ->  1   | 1
|0↓>  -> -1   | 2   
|↑0>  ->  1   | 3
|↑↑>  ->  2   | 4
|↑↓>  ->  0   | 5   
|↓0>  -> -1   | 6   
|↓↑>  ->  0   | 7   
|↓↓>  -> -2   | 8   
"""
struct tJbilayerSz <: AbstractSiteType 
    ifZ2::Bool
end
tJbilayerSz() = tJbilayerSz(false)
export tJbilayerSz
function indextoqn(::tJbilayerSz, i::Int)
    i -= 1
    i == 0 && return 0

    qni = []
    while i > 0
        remainder = i % 9
        if remainder in [8] 
            remainder = -2
        elseif remainder in [2,6] 
            remainder = -1
        elseif remainder in [5,7]
            remainder = 0
        elseif remainder in [1,3]
            remainder = 1
        elseif remainder in [4]
            remainder = 2
        end
        pushfirst!(qni, remainder)
        i = div(i, 9)
    end

    return sum(qni)
end