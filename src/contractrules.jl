using OMEinsum
using OMEinsumContractionOrders
using Random

function generate_vertical_rules(;D=2,χ=20)
	eincode = EinCode(((1,2,3,4,5),# T1
	(6,7,8,9,10),#T2 (dag)

	(11,6,2,12), #swapgate(nl,nu)
	(14,13,8,10), #swapgate(nf,nu)
	(3,16,15,13), #swapgate(nf,nr)
	(4,18,17,16), #swapgate(nl,nu)

	(17,19,20,21,22), #T4
	(23,24,25,26,27), #T3 (dag)

	(21,29,28,27),#swapgate(nl,nu)
	(20,31,30,19),#swapgate(nf,nr)
	(32,33,25,31),#swapgate(nf,nr)
	(23,34,9,33), #swapgate(nl,nu)

	(35,12,1,36), # ACu: E3
	(36,18,5,37), # FRu: E8
	(37,29,22,38), # FRo: E4
	(39,26,28,38),# ACd: E6
	(40,24,34,39), # FLo: E1
	(35,7,11,40) # FLu: E7
	),
	(15,30,14,32) #hamiltonian (ij di dj)
	)
		
	size_dict = [D for i = 1:40]
	size_dict[[3;8;14;15;30;32;20;25]] .= 4
	size_dict[35:40] .= χ
	sd = Dict(i=>size_dict[i] for i = 1:40)

	# for seed =40:100
	seed = 60
	Random.seed!(60)
	optcode = optimize_tree(eincode,sd; sc_target=28, βs=0.1:0.1:10, ntrials=2, niters=100, sc_weight=3.0)


	print("Vertical Contraction Complexity(seed=$(seed))",OMEinsum.timespace_complexity(optcode,sd),"\n") 
	# You would better try some times to make it optimal (By the for-end iteration...)
	# end
	return optcode
end
# const VERTICAL_RULES = generate_vertical_rules()

function generate_horizontal_rules(;D=2,χ=20)
    eincode = EinCode(((1,2,3,4,5),# T1
    (3,4,21,22),#swapgate(nf,nu)

	(6,7,8,9,10),#T2 (dag)

    (23,22,8,27),#swapgate(nf,nu)
    (17,27,10,40), #swapgate(nl,nu)
    (29,6,2,30), #swapgate(nl,nu)

    (16,17,18,19,20),# T4(dag)

    (25,16,18,26),#swapgate(nf,nu)
    (13,26,24,28),#swapgate(nf,nu)
    (5,28,12,33),#swapgate(nl,nu)
    (11,12,13,14,15),# T3

    (31,14,20,32),#swapgate(nl,nu)

    (39,7,29,38), #E1 FLo
    (39,30,1,34), #E2 ACu
    (34,33,11,35), #E3 ARu
    (35,31,15,36), #E4 FRo
    (37,19,32,36), #E5 ARd
    (38,9,40,37), #E6 ACd
	),
	(21,24,23,25) #hamiltonian (ij di dj)
	)

    size_dict = [D for i = 1:40]
	size_dict[[3;8;13;18;21;23;24;25]] .= 4
	size_dict[34:39] .= χ
	sd = Dict(i=>size_dict[i] for i = 1:40)
	
	# for seed = 1:100
	seed = 4
	Random.seed!(seed)
	optcode = optimize_tree(eincode,sd; sc_target=28, βs=0.1:0.1:10, ntrials=3, niters=100, sc_weight=4.0)
	print("Horizontal Contraction Complexity(seed=$(seed))",OMEinsum.timespace_complexity(optcode,sd),"\n")
	# end

	return optcode
end
# const HORIZONTAL_RULES = generate_horizontal_rules()
