

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using FastAI
using ReTest
FastAI.runtests([ReTest.fail, ReTest.not(ReTest.pass)])

module FastAITests

using InlineTest

using ..FastAI
import ..FastAI: Block, Encoding, encodedblock, decodedblock, encode, decode,
    testencoding, test_task_show, checkblock
using ..FastAI.Tabular: EncodedTableRow
using Flux.Optimise: Optimiser, ADAM, apply!
import Makie

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
include("testdata.jl")


include("encodingapi.jl")
include("fasterai.jl")
include("training.jl")

include("makie.jl")

end

FastAITests.runtests([ReTest.fail, ReTest.not(ReTest.pass)])
