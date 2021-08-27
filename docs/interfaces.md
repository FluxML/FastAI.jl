# Interfaces

FastAI.jl provides many interfaces that allow extending its functionality. 

## Learning method interfaces

Learning methods form the core of FastAI.jl's high-level API. See [this tutorial](learning_methods.md) for a motivation and introduction.

Functions for the learning method interfaces always dispatch on a [`LearningMethod`](#). A `LearningMethod` defines everything that needs to happen to turn an input into a target and much more. `LearningMethod` should be a `struct` containing configuration.

### Core interface

Enables training and prediction. Prerequisite for other, optional learning method interfaces.

{.tight}
- Required methods:
    - [`encode`](#) or both [`encodeinput`](#) and [`encodetarget`](#).
    - [`decodeŷ`](#)
- Optional methods:
    - [`shouldbatch`](#)
    - [`encode!`](#) or both [`encodeinput!`](#) and [`encodetarget!`](#).
- Enables use of:
    - [`methoddataset`](#)
    - [`methoddataloaders`](#)
    - [`predict`](#)
    - [`predictbatch`](#)

### Plotting interface

For visualizing observations and predictions using [Makie.jl](https://github.com/JuliaPlots/Makie.jl).

{.tight}
- Required methods:
    - [`plotsample!`](#)
    - [`plotxy!`](#)
    - [`plotprediction!`](#)
- Enables use of:
    - [`plotsamples`](#)
    - [`plotbatch`](#)

### Training interface

Convenience for creating [`Learner`](#)s.

{.tight}
- Required methods:
    - [`methodlossfn`](#)
    - [`methodmodel`](#)
- Enables use of:
    - [`methodlearner`](#)


### Testing interface

Automatically test interfaces.

{.tight}
- Required methods: 
    - [`mockmodel`](#)
    - [`mocksample`](#) or both [`mockinput`](#) and [`mocktarget`](#)
- Enables use of:
    - [`checkmethod_core`](#)


## Callback interface

See the [FluxTraining.jl tutorial](https://lorenzoh.github.io/FluxTraining.jl/dev/docs/callbacks/custom.html).

## Data container interface

