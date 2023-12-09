@with_kw struct SU <: Algorithm
    dτ::Float64 = 0.4
    tratio::Float64 = 0.7
    Mindτ::Float64 = 0.0001
    NoUp::Int64 = 1000
    doEstimate::Bool = true
    tolerance_Es::Float64 = 1.0e-8
    count_upper::Int64 = 200
    count_lower::Int64 = 100
end

function initλΓ(SD, D, d)
    λ = [Iinitial(SD, D; dir = [-1,1]) for _ in 1:4]
    Γ = [randinitial(SD, d, D, D, D, D; dir = [1, -1, -1, 1, 1]) for _ in 1:2]
    normalize!(Γ)
    return λ, Γ
end 

function evoGate(SD, model, dτ)
    h = SD.atype{SD.dtype}.(hamiltonian(model))
 
    U_local = exp(-0.5*dτ*h[2]) 
    function gen_U_2sites( h, dτ )
        d = size(h)[1]
        temp = reshape(h, d^2,d^2)
        temp = exp(-1.0 * dτ * temp) 
        temp = reshape(temp, d,d,d,d)
        return temp
    end
    U_2sites_1 = gen_U_2sites(h[1], 0.5*dτ) 
    
    U_2sites_1 = asSymmetryArray(U_2sites_1, SD.symmetry, SD.stype; dir = [-1,-1,1,1]) 
    U_local = asSymmetryArray(U_local, SD.symmetry, SD.stype; dir = [-1,1])   
    
    return U_local, U_2sites_1
end