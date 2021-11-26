using ADFPEPS
using ArgParse
using CUDA
using Random
using CUDA
using Random
using Test
using OMEinsum
using Optim
using Zygote
CUDA.allowscalar(false)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--tol"
            help = "tol error for vumps"
            arg_type = Float64
            default = 1e-10
        "--maxiter"
            help = "max iterition for vumps"
            arg_type = Int
            default = 10
        "--opiter"
            help = "iterition for optimise"
            arg_type = Int
            default = 1000
        "--f_tol"
            help = "tol error for optimise"
            arg_type = Float64
            default = 1e-10
        "--D"
            help = "ipeps virtual bond dimension"
            arg_type = Int
            required = true
        "--chi"
            help = "vumps virtual bond dimension"
            arg_type = Int
            required = true
        "--folder"
            help = "folder for output"
            arg_type = String
            default = "./data/"
        "--t"
            help = "t"
            arg_type = Float64
            required = true
        "--U"
            help = "U"
            arg_type = Float64
            required = true
        "--mu"
            help = "μ"
            arg_type = Float64
            required = true
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    Random.seed!(100)
    D = parsed_args["D"]
    χ = parsed_args["chi"]
    tol = parsed_args["tol"]
    maxiter = parsed_args["maxiter"]
    opiter = parsed_args["opiter"]
    f_tol = parsed_args["f_tol"]
    folder = parsed_args["folder"]
    t = parsed_args["t"]
    U = parsed_args["U"]
    μ = parsed_args["mu"]
    ipeps, key = init_ipeps(Hubbard(t,U,μ); Ni = 2, Nj = 2, atype = CuArray, folder = folder, D=D, χ=χ, tol=tol, maxiter=maxiter)
    optimiseipeps(ipeps, key; f_tol = f_tol, opiter = opiter, verbose = true)
end

main()