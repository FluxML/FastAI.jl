
Base.@kwdef struct LearningMethodRegistry
    methods::Dict{Any,Any} = Dict()
end

"""
    registerlearningmethod!(registry, methodfn, blocktypes)

Register a learning method constructor `methodfn` as compatible with
`blocktypes` in `registry::LearningMethodRegistry`.

`blocks` should be the least specific set of types that a `methodfn`
can handle. `methodfn` needs to have a method that takes concrete block
instances as the only non-keyword argument, i.e. `methodfn(blocks; kwargs...)`.
"""
function registerlearningmethod!(reg::LearningMethodRegistry, f, blocktypes)
	reg.methods[f] = typify(blocktypes)
	return reg
end

"""
	findlearningmethods(blocktypes)
	findlearningmethods(registry, blocktypes)

Find learning methods compatible with block types `TBlocks` in
`registry::LearningMethodRegistry`.

#### Examples

```julia
julia> findlearningmethods((Image, Label))
[ImageClassificationSingle,]

julia> findlearningmethods((Image, Any))
[ImageClassificationSingle, ImageClassificationMulti, ImageSegmentation, ImageKeypointRegression, ...]
```
"""
function findlearningmethods(reg::LearningMethodRegistry, blocktypes=Any)
	return [methodfn for (methodfn, methodblocks) in reg.methods if blocktypesmatch(methodblocks, blocktypes)]
end


function blocktypesmatch(
        BSupported::Type{<:AbstractBlock},
        BWanted::Type{<:AbstractBlock})
    BWanted <: BSupported
end
function blocktypesmatch(B1::Type{<:Tuple}, B2::Type{<:Tuple})
    all(blocktypesmatch(b1, b2) for (b1, b2) in zip(B1.types, B2.types))
end

blocktypesmatch(BSupported::Type, _::Type{Any}) = true
blocktypesmatch(bsupported, bwanted) = blocktypesmatch(typify(bsupported), bwanted)
blocktypesmatch(BSupported::Type, bwanted) = blocktypesmatch(BSupported, typify(bwanted))

@testset "`blocktypesmatch`" begin
    @test blocktypesmatch(FastAI.Image, FastAI.Image{2})
    @test blocktypesmatch(FastAI.Image, FastAI.Image)
    @test !blocktypesmatch(FastAI.Image{2}, FastAI.Image)
    @test blocktypesmatch(Tuple{FastAI.Image}, Tuple{FastAI.Image})
    @test !blocktypesmatch(Tuple{FastAI.Image{2}}, Tuple{FastAI.Image})
    @test blocktypesmatch(Tuple{FastAI.Image{2}}, Any)
    @test blocktypesmatch(FastAI.Image{2}(), FastAI.Image{2})
    @test blocktypesmatch(FastAI.Image, FastAI.Image{2}())
    @test blocktypesmatch((FastAI.Image, FastAI.Label), (FastAI.Image{2}(), FastAI.Label(1:10)))
    @test blocktypesmatch((FastAI.Image, FastAI.Label), (FastAI.Image{2}(), Any))
end


const FASTAI_METHOD_REGISTRY = LearningMethodRegistry()
