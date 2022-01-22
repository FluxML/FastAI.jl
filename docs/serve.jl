using FastAI
import FastAI: Image
import CairoMakie
using Pollen
using FluxTraining
using DLPipelines
import DataAugmentation
using FilePathsBase
using Colors

function serve(lazy=true; kwargs...)
    refmodules = [FastAI, FluxTraining, DLPipelines, DataAugmentation, DataLoaders, FastAI.Datasets]
    project = Pollen.documentationproject(FastAI; refmodules, watchpackage=true, kwargs...)
    Pollen.serve(project, lazy=lazy)
end
serve()

##


#=
project = Pollen.documentationproject(FastAI; refmodules, inlineincludes = false, )
=#
