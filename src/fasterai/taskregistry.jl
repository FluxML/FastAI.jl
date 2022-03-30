
Base.@kwdef struct LearningTaskRegistry
    tasks::Dict{Any,Any} = Dict()
end

"""
    registerlearningtask!(registry, taskfn, blocktypes)

Register a learning task constructor `taskfn` as compatible with
`blocktypes` in `registry::LearningTaskRegistry`.

`blocks` should be the least specific set of types that a `taskfn`
can handle. `taskfn` needs to have a task that takes concrete block
instances as the only non-keyword argument, i.e. `taskfn(blocks; kwargs...)`.
"""
function registerlearningtask!(reg::LearningTaskRegistry, f, blocktypes)
	reg.tasks[f] = typify(blocktypes)
	return reg
end

"""
	findlearningtasks(blocktypes)
	findlearningtasks(registry, blocktypes)

Find learning tasks compatible with block types `TBlocks` in
`registry::LearningTaskRegistry`.

#### Examples

```julia
julia> findlearningtasks((Image, Label))
[ImageClassificationSingle,]

julia> findlearningtasks((Image, Any))
[ImageClassificationSingle, ImageClassificationMulti, ImageSegmentation, ImageKeypointRegression, ...]
```
"""
function findlearningtasks(reg::LearningTaskRegistry, blocktypes=Any)
	return [taskfn for (taskfn, taskblocks) in reg.tasks if blocktypesmatch(taskblocks, blocktypes)]
end


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
    @test blocktypesmatch((FastAI.Image{2}(), AbstractBlock), (FastAI.Image, FastAI.Label))
    @test !blocktypesmatch(FastAI.Image{2}(), Label(1:10))
end


const FASTAI_METHOD_REGISTRY = LearningTaskRegistry()
