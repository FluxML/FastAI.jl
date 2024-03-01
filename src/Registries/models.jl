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


# ## `ModelVariant` interface
"""
    abstract type ModelVariant

A `ModelVariant` handles loading a model, optionally with pretrained weights and
transforming it so that it can be used for specific learning tasks.


are subblocks (see [`issubblock`](#)) of `blocks = (xblock, yblock)`.

## Interface

- [`compatibleblocks`](#)`(variant)` returns a tuple `(xblock, yblock)` of [`BlockLike`](#) that
    are compatible with the model. This means that a variant can be used for a task with
    input and output blocks `blocks`, if [`issubblock`](#)`(blocks, compatibleblocks(variant))`.
- [`loadvariant`](#)`(::ModelVariant, xblock, yblock, checkpoint; kwargs...)` loads a model
    compatible with block instances `xblock` and `yblock`, with (optionally) weights
    from `checkpoint`.
"""
abstract type ModelVariant end

"""
    compatibleblocks(::ModelVariant)

Indicate compatible input and output block for a model variant.
"""
function compatibleblocks end

function loadvariant end


# ## Model registry creation

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
              loadfn = Field(Any,
                             name = "Load function",
                             optional = false,
                             description = "A function `loadfn(checkpoint)` that loads a default version of the model, possibly with `checkpoint` weights.",
                            ),
              variants = Field(Vector{Pair{String, ModelVariant}},
                               name = "Variants",
                               default = Pair{String, ModelVariant}[],
                               description = "Model variants suitable for different learning tasks. See `?ModelVariant` for more details.",
                               formatfn = d -> join(first.(d), ", ")),
              checkpoints = Field(Vector{String};
                                  name = "Checkpoints",
                                  description = """
                                      Pretrained weight checkpoints that can be loaded for the model. Checkpoints are listed as a
                                      `Vector{String}` and `loadfn` should take care of loading the selected checkpoint""",
                                  formatfn = cs -> join(cs, ", "),
                                  defaultfn = (row, key) -> String[]),
    )
    return Registry(fields; name, loadfn = _loadmodel, description = description)
end

"""
    _loadmodel(row)

Load a model specified by `row` from a model registry.
"""
function _loadmodel(row; input = Any, output = Any, variant = nothing, checkpoint = nothing,
                    pretrained = !isnothing(checkpoint), kwargs...)
    checkpoints, variants = row.checkpoints, row.variants  # 1.6 support

    # Finding matching configuration
    checkpoint = _findcheckpoint(checkpoints; pretrained, name = checkpoint)
    if (pretrained && isnothing(checkpoint))
        throw(NoCheckpointFoundError(checkpoints, checkpoint))
    end

    # If no variant is asked for, use the base model loading function that only takes
    # care of the checkpoint.
    if isnothing(variant) && input === Any && output === Any
        return row.loadfn(checkpoint)
    # If a variant is specified, either by name (through `variant`) or through block
    # constraints `input` or `output`, try to find a matching variant.
    # care of the checkpoint.
    else
        variant = _findvariant(variants, variant, input, output)
        isnothing(variant) && throw(NoModelVariantFoundError(variants, input, output, variant))
        return loadvariant(variant, input, output, checkpoint; kwargs...)
    end
end

# ### Errors

# TODO: Implement Base.showerror
struct NoModelVariantFoundError <: Exception
    variants::Vector
    input::BlockLike
    output::BlockLike
    variant::Union{String, Nothing}
end

# TODO: Implement Base.showerror
struct NoCheckpointFoundError <: Exception
    checkpoints::Vector{String}
    checkpoint::Union{String, Nothing}
end

# ## Create the default registry instance

const MODELS = _modelregistry()

"""
    models()

$_MODELS_DESCRIPTION
"""
models(; kwargs...) = isempty(kwargs) ? MODELS : filter(MODELS; kwargs...)

# ## Helpers

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

function _findvariant(variants::Vector,
                      variantname::Union{String, Nothing}, xblock, yblock)
    if !isnothing(variantname)
        variants = filter(variants) do (name, _)
            name == variantname
        end
    end
    i = findfirst(variants) do (_, variant)
        v_xblock, v_yblock = compatibleblocks(variant)
        issubblock(v_xblock, xblock) && issubblock(v_yblock, yblock)
    end
    isnothing(i) ? nothing : variants[i][2]
end

# ## Tests

struct MockVariant <: ModelVariant
    model
    blocks
end

compatibleblocks(variant::MockVariant) = variant.blocks
loadvariant(variant::MockVariant, _, _, ch) = (ch, variant.model)

@testset "Model registry" begin
    @testset "Basic" begin
        @test_nowarn _modelregistry()
        reg = _modelregistry()
        push!(reg, (; id = "test", loadfn = (checkpoint,) -> checkpoint))

        @test load(reg["test"]) === nothing
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
                                "base" => MockVariant(1, (Any, Any)),
                                "ext" => MockVariant(2, (Any, Label)),
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
            "1" => MockVariant(1, (Any, Any)),
            "2" => MockVariant(1, (Any, Label)),
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
