# Glossary

Terms commonly used in *FastAI.jl*

## Type abbreviations

In many docstrings, generic types are abbreviated with the following symbols. Many of these refer to a learning method; the context should make clear which method is meant.

- `DC{T}`: A data container of type T, meaning a type that implements the data container interface `getobs` and `nobs` where `getobs : (DC{T}, Int) -> Int`, that is, each observation is of type `T`.
- `I`: Type of the unprocessed input in the context of a method.
- `T`: Type of the target variable.
- `X`: Type of the processed input. This is fed into a `model`, though it may be batched beforehand. `Xs` represents a batch of processed inputs.
- `Y`: Type of the model output. `Ys` represents a batch of model outputs.
- `model`/`M`: A learnable mapping `M : (X,) -> Y` or `M : (Xs,) -> Ys`. It predicts an encoded target from an encode input. The learnable part of a learning method.

Some examples of these in use:

- `LearningTask` represents the task of learning to predict `T` from `I`.
- `LearningMethod` is a concrete approach to learning to predict `T` from `I` by using the encoded representations `X` and `Y`.
- `encodeinput : (method, context, I) -> X` encodes an input so that a prediction can be made by a model.
- A task dataset is a `DC{(I, T)}`, i.e. a data container where each observation is a 2-tuple of an input and a target.

## Definitions

### Data container

A data structure that is used to load a number of data observations separately and lazily. It defines how many observations it holds with `nobs` and how to load a single observation with `getobs`.

### Learning method

An instance of `DLPipelines.LearningMethod`. A concrete approach to solving a learning task. Encapsulates the logic and configuration for processing data to train a model and make predictions.

See the DLPipelines.jl documentation for more information. 

### Learning task

An abstract subtype of `DLPipelines.LearningTask` that represents the problem of learning a mapping from some input type `I` to a target type `T`. For example, `ImageClassificationTask` represents the task of learning to map an image to a class. See [learning method](#learning-method)

### Task data container / dataset

`DC{(I, T)}`. A data container containing pairs of inputs and targets. Used in [`methoddataset`](#), [`methoddataloaders`](#) and [`evaluate`](#)