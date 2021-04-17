using Pollen
using FastAI
using FluxTraining
using DLPipelines
using DataAugmentation
using FilePathsBase

refmodules = [FluxTraining, DLPipelines, DataAugmentation, FastAI.Datasets, FastAI]
project = Pollen.documentationproject(FastAI; refmodules, inlineincludes = false)
Pollen.serve(project)
