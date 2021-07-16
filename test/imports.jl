
using Colors: RGB, N0f8, Gray
using FastAI
using FastAI: ParamGroups, IndexGrouper, getgroup, DiscriminativeLRs, decay_optim
using FastAI: Image, Keypoints, Mask, testencoding, Label, OneHot, ProjectiveTransforms,
    encodedblock, decodedblock, encode, decode, mockblock
using FilePathsBase
using FastAI.Datasets
using DLPipelines
import DataAugmentation
import DataAugmentation: getbounds
using Flux
using Flux.Optimise: Optimiser, apply!
using StaticArrays
using Test
using TestSetExtensions
using DataFrames
using Tables
using CSV

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
include("testdata.jl")
