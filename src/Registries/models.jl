# # Model registry
#
# This file defines [`models`](#), a feature registry for models.

# ## Registry definition

const _MODELS_DESCRIPTION = """
A `FeatureRegistry` for models. Allows you to find and load models for various learning
tasks using a unified interface. Call `models()` to see a table view of available models:

```julia
using FastAI
models()
```

Which models are available depends on the loaded packages. For example, FastVision.jl adds
vision models from Metalhead to the registry. Index the registry with a model ID to get more
information about that model:

```julia
using FastAI: models
using FastVision  # loading the package extends the list of available models

models()["metalhead/resnet18"]
```

If you've selected a model, call `load` to then instantiate a model:

```julia
model = load("metalhead/resnet18")
```

By default, `load` loads a default version of the model without any pretrained weights.

`load(model)` also accepts keyword arguments that allow you to specify variants of the model and
weight checkpoints that should be loaded.

Loading a checkpoint of pretrained weights:

- `load(entry; pretrained = true)`: Use any pretrained weights, if they are available.
- `load(entry; checkpoint = "checkpoint-name")`: Use the weights with given name. See
    `entry.checkpoints` for available checkpoints (if any).
- `load(entry; pretrained = false)`: Don't use pretrained weights

Loading a model variant for a specific task:

- `load(entry; input = ImageTensor, output = OneHotLabel)`: Load a model variant matching
    an input and output block.
- `load(entry; variant = "backbone"): Load a model variant by name. See `entry.variants` for
    available variants.
"""

"""
    struct ModelVariant(; transform, xblock, yblock)

A `ModelVariant` is a model transformation that changes a model so that its input and output
are subblocks (see [`issubblock`](#)) of `blocks = (xblock, yblock)`.

The model transformation function `transform` takes a model and two concrete _instances_
of the variant's compatible blocks, returning a transformed model.

    `transform(model, xblock, yblock)`

- `model` is the original model that is transformed
- `xblock` is the [`Block`](#) of the data that is input to the model.
- `yblock` is the [`Block`](#) of the data that the model outputs.

If you're working with a [`SupervisedTask`](#) `task`, these blocks correspond to
`inputblock = getblocks(task).x` and `outputblock = getblocks(task).y`
"""
struct ModelVariant
    transformfn::Any  # callable
    xblock::BlockLike
    yblock::BlockLike
end
_default_transform(model, xblock, yblock; kwargs...) = model
function ModelVariant(; transform = _default_transform, xblock = Any, yblock = Any)
    ModelVariant(transform, xblock, yblock)
end

function _modelregistry(; name = "Models", description = _MODELS_DESCRIPTION)
    fields = (;
              id = Field(String; name = "ID", formatfn = FeatureRegistries.string_format),
              description = Field(String;
                                  name = "Description",
                                  optional = true,
                                  description = "More information about the model",
                                  formatfn = FeatureRegistries.md_format),
              backend = Field(Symbol,
                              name = "Backend",
                              default = :flux,
                              description = "The backend deep learning framework that the model uses. The default is `:flux`."),
              variants = Field(Vector{Pair{String, ModelVariant}},
                               name = "Variants",
                               optional = false,
                               description = "Model variants suitable for different learning tasks. See `?ModelVariant` for more details.",
                               formatfn = d -> join(first.(d), ", ")),
              checkpoints = Field(Vector{String};
                                  name = "Checkpoints",
                                  description = """
                                      Pretrained weight checkpoints that can be loaded for the model. Checkpoints are listed as a
                                      `Vector{String}` and `loadfn` should take care of loading the selected checkpoint""",
                                  formatfn = cs -> join(cs, ", "),
                                  defaultfn = (row, key) -> String[]),
              loadfn = Field(Any;
                             name = "Load function",
                             description = """
                                 Function that loads the base version of the model, optionally with weights.
                                 It is called with the name of the selected checkpoint fro `checkpoints`,
                                 i.e. `loadfn(checkpoint)`. If no checkpoint is selected, it is called with
                                 `nothing`, i.e.  loadfn(`nothing`).

                                 Any unknown keyword arguments passed to `load`, i.e.
                                 `load(registry[id]; kwargs...)` will be passed along to `loadfn`.
                                 """,
                             optional = false))
    return Registry(fields; name, loadfn = _loadmodel, description = description)
end

"""
    _loadmodel(row)

Load a model specified by `row` from a model registry.
"""
function _loadmodel(row; input = Any, output = Any, variant = nothing, checkpoint = nothing,
                    pretrained = !isnothing(checkpoint), kwargs...)
    loadfn, checkpoints, variants = row.loadfn, row.checkpoints, row.variants  # 1.6 support

    # Finding matching configuration
    checkpoint = _findcheckpoint(checkpoints; pretrained, name = checkpoint)

    pretrained && isnothing(checkpoint) &&
        throw(NoCheckpointFoundError(checkpoints, checkpoint))
    variant = _findvariant(variants, variant, input, output)
    isnothing(variant) && throw(NoModelVariantFoundError(variants, input, output, variant))

    # Loading
    basemodel = loadfn(checkpoint, kwargs...)
    model = variant.transformfn(basemodel, input, output)

    return model
end

# ### Errors
struct NoModelVariantFoundError <: Exception
    variants::Vector{Pair{String, ModelVariant}}
    input::BlockLike
    output::BlockLike
    variant::Union{String, Nothing}
end

struct NoCheckpointFoundError <: Exception
    checkpoints::Vector{String}
    checkpoint::Union{String, Nothing}
end

const MODELS = _modelregistry()

"""
    models()

$_MODELS_DESCRIPTION
"""
models(; kwargs...) = isempty(kwargs) ? MODELS : filter(MODELS; kwargs...)

function _findcheckpoint(checkpoints::AbstractVector; pretrained = false, name = nothing)
    if isempty(checkpoints)
        nothing
    elseif !isnothing(name)
        i = findfirst(==(name), checkpoints)
        isnothing(i) ? nothing : checkpoints[i]
    elseif pretrained
        first(values(checkpoints))
    else
        nothing
    end
end

function _findvariant(variants::Vector{Pair{String, ModelVariant}},
                      variantname::Union{String, Nothing}, xblock, yblock)
    if !isnothing(variantname)
        variants = filter(variants) do (name, _)
            name == variantname
        end
    end
    i = findfirst(variants) do (_, variant)
        issubblock(variant.xblock, xblock) && issubblock(variant.yblock, yblock)
    end
    isnothing(i) ? nothing : variants[i][2]
end

# ## Tests

@testset "Model registry" begin
    @testset "Basic" begin
        @test_nowarn _modelregistry()
        reg = _modelregistry()
        push!(reg, (;
                    id = "test",
                    loadfn = _ -> 1,
                    variants = ["base" => ModelVariant()]))

        @test load(reg["test"]) == 1
        @test_throws NoCheckpointFoundError load(reg["test"], pretrained = true)
    end

    @testset "_loadmodel" begin
        reg = _modelregistry()
        @test_nowarn push!(reg,
                           (;
                            id = "test",
                            loadfn = (checkpoint; kwarg = 1) -> (checkpoint, kwarg),
                            checkpoints = ["checkpoint", "checkpoint2"],
                            variants = [
                                "base" => ModelVariant(),
                                "ext" => ModelVariant(((ch, k), i, o; kwargs...) -> (ch,
                                                                                     k + 1),
                                                      Any, Label),
                            ]))
        entry = reg["test"]
        @test _loadmodel(entry) == (nothing, 1)
        @test _loadmodel(entry; pretrained = true) == ("checkpoint", 1)
        @test _loadmodel(entry; checkpoint = "checkpoint2") == ("checkpoint2", 1)
        @test_throws NoCheckpointFoundError _loadmodel(entry; checkpoint = "checkpoint3")

        @test _loadmodel(entry; output = Label) == (nothing, 2)
        @test _loadmodel(entry; variant = "ext") == (nothing, 2)
        @test _loadmodel(entry; pretrained = true, output = Label) == ("checkpoint", 2)
        @test_throws NoModelVariantFoundError _loadmodel(entry; input = Label)
    end

    @testset "_findvariant" begin
        vars = [
            "1" => ModelVariant(identity, Any, Any),
            "2" => ModelVariant(identity, Any, Label),
        ]
        # no restrictions => select first variant
        @test _findvariant(vars, nothing, Any, Any) == vars[1][2]
        # name => select named variant
        @test _findvariant(vars, "2", Any, Any) == vars[2][2]
        # name not found => nothing
        @test _findvariant(vars, "3", Any, Any) === nothing
        # restrict block => select matching
        @test _findvariant(vars, nothing, Any, Label) == vars[2][2]
        # restrict block not found => nothing
        @test _findvariant(vars, nothing, Any, LabelMulti) === nothing
    end

    @testset "_findcheckpoint" begin
        chs = ["check1", "check2"]
        @test _findcheckpoint(chs) === nothing
        @test _findcheckpoint(chs, pretrained = true) === "check1"
        @test _findcheckpoint(chs, pretrained = true, name = "check2") === "check2"
        @test _findcheckpoint(chs, pretrained = true, name = "check3") === nothing
    end
end
