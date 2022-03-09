"""
    abstract type LearningTask

Represents a concrete approach for solving a learning task.

A `LearningTask` defines how data is processed encoded and decoded
before and after going through a model.


## Extending

It is recommended to use [`AbstractBlockTask`](#)s like [`BlockTask`](#)
and [`SupervisedTask`](#) to construct tasks, but you may subtype
`LearningTask` for lower-level control.

There is a core interface that will allow you to train models and
perform inference (for supervised tasks). It consists of

- [`encodesample`](#)
- [`encodeinput`](#)
- [`decodeypred`](#)

You can optionally implement additional interfaces to get support for
higher-level features of the library.

- Training interface: [`tasklossfn`](#), [`taskmodel`](#)
- Testing interface: [`mocksample`](#), [`mockinput`](#), [`mocktarget`](#),
    [`mockmodel`](#)
- Batching: [`shouldbatch`](#)

"""
abstract type LearningTask end


# ## Encoding contexts

"""
    abstract type Context

Represents a context in which a data transformation
is made. This allows using dispatching for varying behavior,
for example, to apply augmentations only during training or
use non-destructive cropping during inference.

See [`Training`](#), [`Validation`](#) and
[`Inference`](#).
"""
abstract type Context end

"""
    Training <: Context

A context for applying data transformations during training. [`Encoding`](#)s and
[`LearningTask`](#)s can dispatch on this when certain transformations,
like random augmentations, should only be applied during training.
"""
struct Training <: Context end

"""
    Validation <: Context

A context for applying data transformations during validation/testing.
[`Encoding`](#)s and [`LearningTask`](#)s can dispatch on this when
certain transformations, like random augmentations, should not be applied
during validation, only in training.
"""
struct Validation <: Context end

struct Inference <: Context end


# ## Encoding interface

function encodesample end

function encodeinput end

function decodeypred end
const decodeŷ = decodeypred

decodey(args...; kwargs...) = decodeypred(args...; kwargs...)

# ## Buffered encoding interface
#
# If not overwritten, applies the non-buffering method.

encodesample!(buf, task, ctx, sample) = encodesample(task, ctx, sample)
encodeinput!(buf, task, ctx, sample) = encodeinput(task, ctx, sample)
decodeypred!(buf, task, ctx, ypred) = decodeypred(task, ctx, ypred)
const decodeŷ! = decodeypred!
decodey!(args...; kwargs...) = decodeypred!(args...; kwargs...)


# ## Training interface

"""
    tasklossfn(task)

Default loss function to use when training models for `task`.
"""
function tasklossfn end


"""
    taskmodel(task, backbone)

Construct a model for `task` from a backbone architecture, for example
by attaching a task-specific head model.
"""
function taskmodel end


# ## Testing interface

"""
    mocksample(task)

Generate a random `sample` compatible with `task`.
"""
mocksample(task) = (mockinput(task), mocktarget(task))

"""
    mockinput(task)

Generate a random `input` compatible with `task`.
"""
function mockinput end

"""
    mocktarget(task)

Generate a random `target` compatible with `task`.
"""
function mocktarget end

"""
    mockmodel(task)

Generate a `model` compatible with `task` for testing.
"""
function mockmodel end

# ## Batching interface



"""
    shouldbatch(::LearningTask)

Define whether encoded samples for a learning task should be
batched. The default is `true`.
"""
shouldbatch(::LearningTask) = true
