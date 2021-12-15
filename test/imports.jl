
using FastAI
import FastAI: Block, Encoding, encodedblock, decodedblock, encode, decode,
    testencoding, test_method_show, checkblock
using FastAI.Tabular: EncodedTableRow
using Flux.Optimise: Optimiser, ADAM, apply!
import Makie

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
include("testdata.jl")
