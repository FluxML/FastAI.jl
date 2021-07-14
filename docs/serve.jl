using CairoMakie
using Pollen
using FastAI
using FluxTraining
using DLPipelines
using DataAugmentation
using FilePathsBase
using Colors

refmodules = [FastAI, FluxTraining, DLPipelines, DataAugmentation, FastAI.Datasets]
project = Pollen.documentationproject(FastAI; refmodules, watchpackage=true)

##

Pollen.serve(project)

#=
project = Pollen.documentationproject(FastAI; refmodules, inlineincludes = false, )
=#
