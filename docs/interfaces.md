# Interfaces

FastAI.jl provides many interfaces that allow extending its functionality. 

## Learning task interfaces

Learning tasks form the core of FastAI.jl's high-level API. See [this tutorial](learning_tasks.md) for a motivation and introduction.

Functions for the learning task interfaces always dispatch on a [`LearningTask`](#). A `LearningTask` defines everything that needs to happen to turn an input into a target and much more. `LearningTask` should be a `struct` containing configuration.

### Core interface

Enables training and prediction. Prerequisite for other, optional learning task interfaces.

{.tight}
- Required tasks:
    - [`encode`](#) or both [`encodeinput`](#) and [`encodetarget`](#).
    - [`decodeŷ`](#)
- Optional tasks:
    - [`shouldbatch`](#)
    - [`encode!`](#) or both [`encodeinput!`](#) and [`encodetarget!`](#).
- Enables use of:
    - [`taskdataset`](#)
    - [`taskdataloaders`](#)
    - [`predict`](#)
    - [`predictbatch`](#)

### Plotting interface

For visualizing observations and predictions using [Makie.jl](https://github.com/JuliaPlots/Makie.jl).

### Training interface

Convenience for creating [`Learner`](#)s.

{.tight}
- Required tasks:
    - [`tasklossfn`](#)
    - [`taskmodel`](#)
- Enables use of:
    - [`tasklearner`](#)


### Testing interface

Automatically test interfaces.

{.tight}
- Required tasks: 
    - [`mockmodel`](#)
    - [`mocksample`](#) or both [`mockinput`](#) and [`mocktarget`](#)
- Enables use of:
    - [`checktask_core`](#)


## Callback interface

See the [FluxTraining.jl tutorial](https://lorenzoh.github.io/FluxTraining.jl/dev/docs/callbacks/custom.html).

## Data container interface

