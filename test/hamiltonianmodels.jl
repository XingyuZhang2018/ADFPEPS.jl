using ADFPEPS
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
    println(hamiltonian(Hubbard(1.0,0.0,0.0)))                                                                                                                                                                                           
end

@testset "hop_pair" begin
    @test hop_pair(1.0,1.0) isa HamiltonianModel 
    model = hop_pair(rand(),rand()) 
    @test hamiltonian(model) == hamiltonian(model)'                                                                                                                
end