module Registries

using ..FastAI
using ..FastAI.Datasets
using ..FastAI.Datasets: DatasetLoader, DataDepLoader, isavailable, loaddata, typify

import Markdown
using DataDeps
using FeatureRegistries
using FeatureRegistries: Registry, Field
using InlineTest


_formatblock(t::Type{<:Tuple}) = _formatblock(Tuple(t.types))
_formatblock(t::Tuple) = map(_formatblock, t)
_formatblock(T::Type) = T
_formatblock(::T) where T = T

function blocktypesmatch(
        BSupported::Type,
        BWanted::Type)
    # true if both types are part of the same type tree
    return BSupported <: BWanted || BWanted <: BSupported
end
function blocktypesmatch(B1::Type{<:Tuple}, B2::Type{<:Tuple})
    all(blocktypesmatch(b1, b2) for (b1, b2) in zip(B1.types, B2.types))
end


blocktypesmatch(BSupported::Type, ::Type{Any}) = true
blocktypesmatch(BSupported::Type{Any}, ::Type) = true
blocktypesmatch(BSupported::Type{Any}, ::Type{Any}) = true
blocktypesmatch(bsupported, bwanted) = blocktypesmatch(typify(bsupported), typify(bwanted))

@testset "`blocktypesmatch`" begin
    @test blocktypesmatch(FastAI.Label, FastAI.Label{String})
    @test blocktypesmatch(FastAI.Label, FastAI.Label)
    @test blocktypesmatch(FastAI.Label{String}, FastAI.Label)
    @test blocktypesmatch(Tuple{FastAI.Label}, Tuple{FastAI.Label})
    @test blocktypesmatch(Tuple{FastAI.Label{String}}, Tuple{FastAI.Label})
    @test blocktypesmatch(Tuple{FastAI.Label{String}}, Any)
    @test blocktypesmatch(FastAI.Label{String}(["x", "y"]), FastAI.Label{String})
    @test blocktypesmatch(FastAI.Label, FastAI.Label{String}(["x", "y"]))
    @test blocktypesmatch((FastAI.Label{String}(["x", "y"]), FastAI.Label(1:10)), (FastAI.Label, FastAI.Label))
    @test blocktypesmatch((FastAI.Label{String}(["x", "y"]), FastAI.AbstractBlock), (FastAI.Label, FastAI.Label))
    @test !blocktypesmatch(FastAI.Label{String}(["x", "y"]), Label(1:10))
end


include("datasets.jl")
include("tasks.jl")
include("recipes.jl")


export datasets,
    learningtasks,
    datarecipes,
    find,
    info,
    load


function __init__()
    _registerdatasets(DATASETS)
end

end
