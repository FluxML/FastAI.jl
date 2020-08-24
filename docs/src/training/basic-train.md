# Basic Training Functionality

Basic training wraps together the data (in a [`DataBunch`](@ref) object) with a Flux model to define a [`Learner`](@ref) object. The `Learner` object is the entry point of most of the [Callback](@ref Callbacks) objects that will customize this training loop in different ways.

## `Learner`

A [`Learner`](@ref) type wraps a [`DataBunch`](@ref) with a model and optimizer. The model can then be [`fit`](@ref) using the optimizer by training it for several epochs. The training process is customized by adding [Callbacks](@ref) to the `Learner`.

```@docs
Learner
add_cb!
FastAI.cbs
data_bunch
FastAI.data_bunch!
model
FastAI.model!
loss
loss!
opt
opt!
```

## `AbstractLearner` Interface

If you want to create a custom learner, you can implement the [`AbstractLearner`](@ref) interface.

```@docs
AbstractLearner
```

## `Recorder`

A [`Recorder`](@ref) logs the values of metrics throughout the training history.

```@docs
Recorder
FastAI.add!(::Recorder, ::Any, ::Any, ::Any)
FastAI.log!
```