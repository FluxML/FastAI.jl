
using Colors: RGB, N0f8, Gray
using FastAI
using FastAI: ParamGroups, IndexGrouper, getgroup, DiscriminativeLRs, decay_optim
import FastAI: Image, Keypoints, Mask, testencoding, Label, OneHot, ProjectiveTransforms,
    encodedblock, decodedblock, encode, decode, mockblock, checkblock, Block, Encoding
using FilePathsBase
using FastAI.Datasets
using FastAI.Models
using DLPipelines
import DataAugmentation
import DataAugmentation: getbounds, NormalizeRow
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
