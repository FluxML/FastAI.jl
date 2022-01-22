
using Pollen
using FastAI
using FastAI.Vision: Image
Image
using FluxTraining
using DLPipelines
import DataAugmentation
using FilePathsBase

import CairoMakie

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

refmodules = [FastAI, FluxTraining, DLPipelines, DataAugmentation, FastAI.Datasets, FastAI.Models]
project = Pollen.documentationproject(FastAI; refmodules = refmodules)
Pollen.fullbuild(project, Pollen.FileBuilder(Pollen.HTML(), p"dev/"))
