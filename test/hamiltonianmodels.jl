using ADFPEPS
using ADFPEPS:Nup,Ndn
using Test
using ITensors
using OMEinsum
using LinearAlgebra

@testset "Hubbard" begin
    @test Hubbard(1.0,1.0,1.0) isa HamiltonianModel
    model = Hubbard(rand(),rand(),rand())
    @test hamiltonian(model) == hamiltonian(model)'    
    @test all(diag(hamiltonian(Hubbard(1.0,0.0,0.0))) .== 0)
    @test diag(hamiltonian(Hubbard(1.0,4.0,0.0))) == 4*1/4* [0.0,0.0,0.0,1.0,
                                                             0.0,0.0,0.0,1.0,
                                                             0.0,0.0,0.0,1.0,
                                                             1.0,1.0,1.0,2.0]

    @test diag(hamiltonian(Hubbard(1.0,0.0,4.0))) == -4*1/4* [0.0,1.0,1.0,2.0,
                                                              1.0,2.0,2.0,3.0,
                                                              1.0,2.0,2.0,3.0,
                                                              2.0,3.0,3.0,4.0]

    @test diag(hamiltonian(Hubbard(1.0,4.0,2.0)) + 4*1/8*I(16)) == 4*1/8*[1.0, 0.0, 0.0,1.0,
                                                                          0.0,-1.0,-1.0,0.0,
                                                                          0.0,-1.0,-1.0,0.0,
                                                                          1.0, 0.0, 0.0,1.0]    
    println(eigen((hamiltonian(Hubbard(1.0,0.0,0.0)))).values)                                                                                                                                                                                         
end

@testset "hop_pair" begin
    @test hop_pair(1.0,1.0) isa HamiltonianModel 
    model = hop_pair(rand(),rand()) 
    @test hamiltonian(model) == hamiltonian(model)'                                                                                                                
end

@testset "Hubbard_hand" begin
    H = Hubbard_hand(Hubbard(rand(),rand(),rand()))
    @test H == H' 
    @test all(diag(Hubbard_hand(Hubbard(1.0,0.0,0.0))) .== 0)
    @test diag(Hubbard_hand(Hubbard(1.0,4.0,0.0))) == 4*1/4* [0.0,0.0,0.0,1.0,
                                                     0.0,0.0,0.0,1.0,
                                                     0.0,0.0,0.0,1.0,
                                                     1.0,1.0,1.0,2.0]

    @test diag(Hubbard_hand(Hubbard(1.0,0.0,4.0))) == -4*1/4* [0.0,1.0,1.0,2.0,
                                                              1.0,2.0,2.0,3.0,
                                                              1.0,2.0,2.0,3.0,
                                                              2.0,3.0,3.0,4.0]

    @test diag(Hubbard_hand(Hubbard(1.0,4.0,2.0)) + 4*1/8*I(16)) == 4*1/8*[1.0, 0.0, 0.0,1.0,
                                                                          0.0,-1.0,-1.0,0.0,
                                                                          0.0,-1.0,-1.0,0.0,
                                                                          1.0, 0.0, 0.0,1.0]                                                                                                                                                                                         
end

@testset "Hubbard_hand" begin
    model = Hubbard(1.0, 12.0, 6.0)
    h1 = reshape(hamiltonian(model), 4, 4, 4, 4)
    h2 = reshape(hamiltonian_hand(model), 4, 4, 4, 4)
    @test h1 == h2                                                                                                     
end

@testset "hop_pair" begin
    model = hop_pair(1.0, 1.0)
    h1 = reshape(hamiltonian(model), 4, 4, 4, 4)
    h2 = reshape(hamiltonian_hand(model), 4, 4, 4, 4)
    @test h1 == h2                                                                                               
end

@testset "observable" begin
    @test hamiltonian(Occupation()) == [0.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 2.0]
    @test hamiltonian(DoubleOccupation()) == [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0]
end

@testset "find U" begin
    # S1 = Hubbard_hand(Hubbard(1.0,0.0,0.0))
    S2 = hamiltonian(Hubbard(1.0,0.0,0.0))
    S3 = hamiltonian(THubbard(1.0,0.0,0.0))
    # for i = 1:16
    #     println(S3[i,:])
    # end
    # U1 = eigen(S1).vectors
    # U2 = eigen(S2).vectors
    # U3 = eigen(S3).vectors
    # U12 = U1*U2'
    # U32 = U3*U2'
    # # U = diagm(ones(16))
    # # for i in [7,14,12]
    # #     U[i,i] = -1.0
    # # end
    U = [1 0 0 0;0 0 1 0;0 -1 0 0;0 0 0 1]
    U32 = reshape(ein"ab,cd -> acbd"(U,I(4)),16,16)
    @test U32'*U32 ≈ I(16)
    # @test U32'≈ U32
    @test S2 ≈ U32'*S3*U32
    # # U32 = map(x->abs(x) < 1e-10 ? 0.0 : x,U32)
    # # @show U
end

@testset "find all -" begin
    for i in [-1.0,1.0], j in [-1.0,1.0], k in [-1.0,1.0], l in [-1.0,1.0]
        H = hamiltonian(THubbard([i,j,k,l],0.0,0.0))
        @show i,j,k,l,sum(H)
    end
end

@testset "tJ" begin
    @test tJ(1.0,1.0) isa HamiltonianModel
    @test hamiltonian(tJ(1.0,1.0)) == hamiltonian_hand(tJ(1.0,1.0))
end