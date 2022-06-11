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
    @test blocktypesmatch(FastAI.Image, FastAI.Image{2})
    @test blocktypesmatch(FastAI.Image, FastAI.Image)
    @test blocktypesmatch(FastAI.Image{2}, FastAI.Image)
    @test blocktypesmatch(Tuple{FastAI.Image}, Tuple{FastAI.Image})
    @test blocktypesmatch(Tuple{FastAI.Image{2}}, Tuple{FastAI.Image})
    @test blocktypesmatch(Tuple{FastAI.Image{2}}, Any)
    @test blocktypesmatch(FastAI.Image{2}(), FastAI.Image{2})
    @test blocktypesmatch(FastAI.Image, FastAI.Image{2}())
    @test blocktypesmatch((FastAI.Image{2}(), FastAI.Label(1:10)), (FastAI.Image, FastAI.Label))
    @test blocktypesmatch((FastAI.Image{2}(), FastAI.AbstractBlock), (FastAI.Image, FastAI.Label))
    @test !blocktypesmatch(FastAI.Image{2}(), Label(1:10))
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
