using CairoMakie
using Pollen
using FastAI
using FluxTraining
using DLPipelines
using DataAugmentation
using FilePathsBase

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

refmodules = [FastAI, FluxTraining, DLPipelines, DataAugmentation, FastAI.Datasets]
project = Pollen.documentationproject(FastAI; refmodules = refmodules)
Pollen.fullbuild(project, Pollen.FileBuilder(Pollen.HTML(), p"dev/"))
