# Basic Training Functionality

Basic training wraps together the data (in a [`DataBunch`](@ref) object) with a Flux model to define a [`Learner`](@ref) object. The `Learner` object is the entry point of most of the [Callback](@ref Callbacks) objects that will customize this training loop in different ways.

<!-- from fastai docs

Some of the most commonly used customizations are available through the train module, notably:

Learner.lr_find will launch an LR range test that will help you select a good learning rate.
Learner.fit_one_cycle will launch a training using the 1cycle policy to help you train your model faster.
Learner.to_fp16 will convert your model to half precision and help you launch a training in mixed precision. -->

## `Learner`

A [`Learner`](@ref) type wraps a [`DataBunch`](@ref) with a model and optimizer. The model can then be [`fit`](@ref) using the optimizer by training it for several epochs. The training process is customized by adding [Callbacks](@ref) to the `Learner`.

```@docs
Learner
add_cb!
cbs
data_bunch
data_bunch!
model
model!
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
add!(::Recorder, ::Any, ::Any, ::Any)
log!
```