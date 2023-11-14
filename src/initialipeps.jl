export initialfromD

function initialfromD(model, folder, Ni, Nj, symmetry, sitetype,
                      oldindD, oldindχ, olddimsD, olddimsχ, 
                      newindD, newindχ, newdimsD, newdimsχ,
                      tol, maxiter, miniter; 
                      sortqn::Bool = false
                      )
    oldD = sum(olddimsD)
    oldχ = sum(olddimsχ)
    newD = sum(newdimsD)
    newχ = sum(newdimsχ)
    oldipeps, key = init_ipeps(model; 
    Ni=Ni, Nj=Nj, symmetry=symmetry, sitetype=sitetype, atype=Array, folder=folder, d = 9, 
    D=oldD, χ=oldχ, indD = oldindD, 
    indχ = oldindχ, dimsD = olddimsD, dimsχ = olddimsχ, 
    tol=tol, maxiter=maxiter, miniter=miniter)

    _, model, Ni, Nj, symmetry, sitetype, atype, _, _, tol, maxiter, _, _, _, _  = key
    oldinfo = zerosinitial(Val(symmetry), Array, ComplexF64, oldD, oldD, 9, oldD, oldD; 
		dir = [-1, -1, 1, 1, 1], 
		indqn = [oldindD, oldindD, getqrange(sitetype, 9)..., oldindD,oldindD],                    
		indims = [olddimsD, olddimsD, getblockdims(sitetype, 9)..., olddimsD, olddimsD], 
		f = [0],
        ifZ2=sitetype.ifZ2
		)
    oldipeps = [U1Array(oldinfo.qn, oldinfo.dir, [reshape(atype(oldipeps[1 + sum(prod.(oldinfo.dims[1:j-1])):sum(prod.(oldinfo.dims[1:j])), i]), tuple(oldinfo.dims[j]...)) for j in 1:length(oldinfo.dims)], oldinfo.size, oldinfo.dims, 1, sitetype.ifZ2) for i = 1:Int(ceil(Ni*Nj/2))]

    newinfo = zerosinitial(Val(symmetry), Array, ComplexF64, newD, newD, 9, newD, newD; 
                    dir = [-1, -1, 1, 1, 1], 
                    indqn = [newindD, newindD, getqrange(sitetype, 9)..., newindD, newindD],                    
                    indims = [newdimsD, newdimsD, getblockdims(sitetype, 9)..., newdimsD, newdimsD], 
                    f = [0],
                    ifZ2=sitetype.ifZ2
                    )
    newipeps = [newinfo.tensor for i = 1:Int(ceil(Ni*Nj/2))]
    newipeps = [U1Array(newinfo.qn, newinfo.dir, [reshape(atype(newipeps[i][1 + sum(prod.(newinfo.dims[1:j-1])):sum(prod.(newinfo.dims[1:j]))]), tuple(newinfo.dims[j]...)) for j in 1:length(newinfo.dims)], newinfo.size, newinfo.dims, 1, sitetype.ifZ2) for i = 1:Int(ceil(Ni*Nj/2))]

    for i = 1:Int(ceil(Ni*Nj/2))
        for j = 1:length(newipeps[i].qn)
            (newipeps[i].tensor)[j] .+= 1e-6 * randn(ComplexF64, (newipeps[i].dims)[j]...)
        end
    end
    for i = 1:Int(ceil(Ni*Nj/2))
        for j = 1:length(oldipeps[i].qn)
            index = indexin([(oldipeps[i].qn)[j]], newipeps[i].qn)
            olddims = (oldipeps[i].dims)[j]
            div = [1:d for d in olddims]
            (newipeps[i].tensor)[index...][div...] = (oldipeps[i].tensor)[j]
        end
    end
    
    newsdims = sum(prod.(newipeps[1].dims))
    newinitial = zeros(ComplexF64, newsdims, Int(ceil(Ni*Nj/2)))
    if sortqn == true
        @show "sortqn"
        p = sortperm(newipeps[1].qn)
        newinitial[:, 1] = vcat(map(vec, newipeps[1].tensor[p])...)
        # newinitial[:, 2] = vcat(map(vec, newipeps[2].tensor[p])...)
    else
        @show "no sortqn"
        newinitial[:, 1] = vcat(map(vec, newipeps[1].tensor)...)
        # newinitial[:, 2] = vcat(map(vec, newipeps[2].tensor)...)
    end

    folder = folder*"/$(model)_$(Ni)x$(Nj)_$(newindD)_$(newdimsD)/"
    save(folder*"D$(newD)_χ$(newχ)_tol$(tol)_maxiter$(maxiter).jld2", "ipeps", newinitial)
end