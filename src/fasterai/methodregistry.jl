
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
	return [methodfn for (methodfn, methodblocks) in reg.methods if typify(blocks) <: methodblocks]
end


const FASTAI_METHOD_REGISTRY = LearningMethodRegistry()
