export initialfromD

function initialfromD(model, folder, Ni, Nj, symmetry, oldD, newD, oldχ, newχ)
    oldipeps, key = init_ipeps(model; Ni=Ni, Nj=Nj, symmetry=symmetry, atype=Array, folder=folder, D=oldD, χ=oldχ, tol=1e-10, maxiter=10)
    folder, model, Ni, Nj, symmetry, atype, D, χ, tol, maxiter = key
    oldinfo = zerosinitial(Val(symmetry), Array, ComplexF64, oldD,oldD,4,oldD,oldD; dir = [-1,-1,1,1,1], q = [1])
    oldipeps = [asArray(U1Array(oldinfo.qn, oldinfo.dir, [reshape(atype(oldipeps[1 + sum(prod.(oldinfo.dims[1:j-1])):sum(prod.(oldinfo.dims[1:j])), ABBA(i)]), tuple(oldinfo.dims[j]...)) for j in 1:length(oldinfo.dims)], oldinfo.size, oldinfo.dims, 1)) for i = 1:Ni*Nj]

    newipeps = [zeros(ComplexF64, newD, newD, 4, newD, newD) for _ = 1:Ni*Nj]
    for i = 1:Ni*Nj
        newipeps[i][1:oldD, 1:oldD, :, 1:oldD, 1:oldD] = oldipeps[i]
    end
    newipeps = [asU1Array(newipeps[i], dir = [-1,-1,1,1,1], q=[1]) for i = 1:Ni*Nj]
    
    newsdims = sum(prod.(newipeps[1].dims))
    newinitial = zeros(ComplexF64, newsdims, Int(ceil(Ni*Nj/2)))
    newinitial[:, 1] = vcat(map(x->x[:], newipeps[1].tensor)...)
    newinitial[:, 2] = vcat(map(x->x[:], newipeps[2].tensor)...)

    save(folder*"$(model)_$(Ni)x$(Nj)/D$(newD)_χ$(newχ)_tol$(tol)_maxiter$(maxiter).jld2", "ipeps", newinitial)
end