using CairoMakie
using Pollen
using FastAI
using FluxTraining
using DLPipelines
using DataAugmentation
using FilePathsBase
using Colors

function serve(lazy=true)
    refmodules = [FastAI, FluxTraining, DLPipelines, DataAugmentation, FastAI.Datasets]
    project = Pollen.documentationproject(FastAI; refmodules, watchpackage=true)
    Pollen.serve(project, lazy=lazy)
end
serve()

##


#=
project = Pollen.documentationproject(FastAI; refmodules, inlineincludes = false, )
=#
