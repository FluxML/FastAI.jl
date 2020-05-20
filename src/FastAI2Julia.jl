#=
FastAI2Julia.jl:

Author: Peter Wolf (opus111@gmail.com)

A first cut at a port of the FastAI V2 API to Julia

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/

The main purpose of this code is to see if the team likes the method
of defining an interface and implementations in Julia
=#

module FastAI2Julia

export fit
export add_cb
export AbstractCallback
export TrainEvalCallback
export Learner

"""
Basic class handling tweaks of the training loop by changing a [Learner](@ref) in various events

The training loop is defined in [Learner](@ref) a bit below and consists in a minimal set of instructions: looping through the data we:

compute the output of the model from the input
calculate a loss between this output and the desired target
compute the gradients of this loss with respect to all the model parameters
update the parameters accordingly
zero all the gradients

Any tweak of this training loop is defined in a Callback to avoid over-complicating the code of the training loop, and to make it easy to mix and match different techniques (since they'll be defined in different callbacks).

A callback can implement the following methods:

begin_fit
after_fit
begin_train
after_train
begin_epoch
after_epoch
begin_batch
after_batch
begin_validate
after_validate
after_pred
after_loss
after_backward
after_step
after_cancel_batch
after_batch

By default handling of these events do nothing.  Special behavior is implemented by overriding these methods

"""
abstract type AbstractCallback end

"""
Group together a model, some dls and a loss_func to handle training

opt_func will be used to create an optimizer when Learner.fit is called, with lr as a default learning rate. splitter is a function that takes learner.model and returns a list of parameter groups (or just one parameter group if there are no different parameter groups). The default is trainable_params, which returns all trainable parameters of the model.

cbs is one or a list of Callbacks [AbstractCallback](@ref) to pass to the Learner. Callbacks are used for every tweak of the training loop. Each Callback is registered as an attribute of Learner (with camel case). At creation, all the callbacks in defaults.callbacks (TrainEvalCallback, Recorder and ProgressCallback) are associated to the Learner.

metrics is an optional list of metrics, that can be either functions or Metrics (see below).

path and model_dir are used to save and/or load models. Often path will be inferred from dls, but you can override it or pass a Path object to model_dir. Make sure you can write in path/model_dir!

wd is the default weight decay used when training the model; moms, the default momentums used in Learner.fit_one_cycle. wd_bn_bias controls if weight decay is applied to BatchNorm layers and bias.

Lastly, train_bn controls if BatchNorm layers are trained even when they are supposed to be frozen according to the splitter. Our empirical experiments have shown that it's the best behavior for those layers in transfer learning.
"""
mutable struct Learner
    cbs:: Array{AbstractCallback}
    opt
    wd
    n_epoch
    loss
    dls
end

include("callback/Callback.jl")
include("learner/Learner.jl")

end
