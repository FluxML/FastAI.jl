
using CUDA
using Colors: RGB
using FastAI
using FilePathsBase
using FastAI.Datasets
using DLPipelines
using DataAugmentation
using DataAugmentation: getbounds
using Flux
using StaticArrays
using Test
using TestSetExtensions

#ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
include("testdata.jl")

function test_gpu(f)
    if CUDA.functional()
        f()
    else
        @test_broken "No GPU"
    end
end
