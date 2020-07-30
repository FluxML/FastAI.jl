#=
learner.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the FastAI V2 Learner API to Julia

Methods for handling the training loop

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/learner.py

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/learner.html

The main purpose of this code is to see if the team likes the method
of defining an interface and implementations in Julia
=#
using Flux
using Infiltrator
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

struct CancelFitException <: Exception end
struct CancelEpochTrainException <: Exception end
struct CancelBatchTrainException <: Exception end
struct CancelEpochValidateException <: Exception end
struct CancelBatchValidateException <: Exception end

"""
Types representing the concept `Learner`.  

Group together a model, some dls and a loss_func to handle training

opt_func will be used to create an optimizer when Learner.fit is called, with lr as a default learning rate. splitter is a function that takes learner.model and returns a list of parameter groups (or just one parameter group if there are no different parameter groups). The default is trainable_params, which returns all trainable parameters of the model.

cbs is one or a list of Callbacks [AbstractCallback](@ref) to pass to the Learner. Callbacks are used for every tweak of the training loop. Each Callback is registered as an attribute of Learner (with camel case). At creation, all the callbacks in defaults.callbacks (TrainEvalCallback, Recorder and ProgressCallback) are associated to the Learner.

metrics is an optional list of metrics, that can be either functions or Metrics (see below).

path and model_dir are used to save and/or load models. Often path will be inferred from dls, but you can override it or pass a Path object to model_dir. Make sure you can write in path/model_dir!

wd is the default weight decay used when training the model; moms, the default momentums used in Learner.fit_one_cycle. wd_bn_bias controls if weight decay is applied to BatchNorm layers and bias.

Lastly, train_bn controls if BatchNorm layers are trained even when they are supposed to be frozen according to the splitter. Our empirical experiments have shown that it's the best behavior for those layers in transfer learning.

In Julia duck typing, implementing an interface just requires 
implementing a set of required fuctions. 

For a type T to be a Learner, the required functions are:

current_batch(learner:: T)  Returns the predictions and targets (y) for the current batch
loss(learner:: T) Returns losses for current batch 
"""
abstract type AbstractLearner end

mutable struct Learner <: AbstractLearner
    cbs:: Array{AbstractCallback}
    db::DataBunch
    model
    opt
    loss
    batch
    batch_validate_loss::Real
end

Learner(data_bunch, model; opt=Flux.ADAM(), loss=Flux.mse) = Learner([],data_bunch,model,opt,loss,nothing,0.0)
data_bunch(l::Learner) = l.db
data_bunch!(l::Learner,data_bunch) = l.db = data_bunch
model(l::Learner) = l.model
model!(l::Learner,model) = l.model=model
loss(l::Learner) = l.loss
loss!(l::Learner) = l.loss=loss
opt(l::Learner) = l.opt
opt!(l::Learner,opt) = l.opt=opt

batch(l::Learner) = l.batch
batch_size(l::Learner) = length(l.batch)
batch_validate_loss(l::Learner) = l.batch_validate_loss

"""
add_cb!(learner::Learner,cb::AbstractCallback cb)

Add a new Callback [AbstractCallback](@ref) to this Learner [Learner](@ref)
"""
add_cb!(learner::Learner,cb::AbstractCallback) = push!(learner.cbs,cb)

# pass event to all callbacks
_cbs_before_fit(learner::Learner, n_epoch) =  for cb in learner.cbs before_fit(cb,learner,n_epoch) end
_cbs_after_fit(learner::Learner) =  for cb in learner.cbs after_fit(cb,learner) end
_cbs_after_cancel_fit(learner::Learner) =  for cb in learner.cbs after_cancel_fit(cb,learner) end

_cbs_before_epoch(learner::Learner, epoch) =  for cb in learner.cbs before_epoch(cb,learner,epoch) end
_cbs_after_epoch(learner::Learner, epoch) =  for cb in learner.cbs after_epoch(cb,learner,epoch) end
_cbs_after_cancel_epoch(learner::Learner, epoch) =  for cb in learner.cbs after_cancel_epoch(cb,learner,epoch) end

_cbs_before_epoch_train(learner::Learner, epoch) =  for cb in learner.cbs before_epoch_train(cb,learner,epoch) end
_cbs_after_epoch_train(learner::Learner, epoch) =  for cb in learner.cbs after_epoch_train(cb,learner,epoch) end
_cbs_after_cancel_epoch_train(learner::Learner, epoch) =  for cb in learner.cbs after_cancel_epoch_train(cb,learner,epoch) end

_cbs_before_batch_train(learner::Learner, batch_index, epoch) =  for cb in learner.cbs before_batch_train(cb,learner, batch_index, epoch) end
_cbs_after_batch_train(learner::Learner, batch_index, epoch) =  for cb in learner.cbs after_batch_train(cb,learner, batch_index, epoch) end
_cbs_after_cancel_batch_train(learner::Learner, batch_index, epoch) =  for cb in learner.cbs after_cancel_batch_train(cb,learner, batch_index, epoch) end

_cbs_before_epoch_validate(learner::Learner, epoch) =  for cb in learner.cbs before_epoch_validate(cb,learner,epoch) end
_cbs_after_epoch_validate(learner::Learner, epoch) =  for cb in learner.cbs after_epoch_validate(cb,learner,epoch) end
_cbs_after_cancel_epoch_validate(learner::Learner, epoch) =  for cb in learner.cbs after_cancel_epoch_validate(cb,learner,epoch) end

_cbs_before_batch_validate(learner::Learner, batch_index, epoch) =  for cb in learner.cbs before_batch_validate(cb,learner, batch_index, epoch) end
_cbs_after_batch_validate(learner::Learner, batch_index, epoch) =  for cb in learner.cbs after_batch_validate(cb,learner, batch_index, epoch) end
_cbs_batch_validate_loss(learner::Learner,loss,batch_index,epoch) =  for cb in learner.cbs batch_validate_loss(cb,learner, loss, batch_index, epoch) end
_cbs_after_cancel_batch_validate(learner::Learner, batch_index) =  for cb in learner.cbs after_cancel_batch_validate(cb,learner, batch_index, epoch) end

function _do_batch_train(learner::Learner, batch, batch_index, epoch, ps)
    try
       #print("_do_batch_train ")
        _loss(xy) = learner.loss(learner.model(xy[1]),xy[2])
        _cbs_before_batch_train(learner,batch_index,epoch)
       #print("a")
        gs = gradient(ps) do
            sum(_loss.(batch))
        end
       #print("b")
        update!(learner.opt, ps, gs)
       #println("c")
    #catch CancelBatchTrainException
    #    _cbs_after_cancel_batch_train(learner,batch_index,epoch)
    finally
        _cbs_after_batch_train(learner,batch_index,epoch)
    end
end

function _do_epoch_train(learner::Learner, epoch)
    try
       #print("_do_epoch_train ")
        data = learner|> data_bunch |> train
        ps = params(learner.model)
       #print("a")
        _cbs_before_epoch_train(learner,epoch)
       #print("b")
        for (i,batch) in enumerate(data)
            _do_batch_train(learner,batch,i,epoch,ps)
        end 
    #catch CancelEpochTrainException
    #    _cbs_after_cancel_epoch_train(learner,epoch)
    finally
        _cbs_after_epoch_train(learner,epoch)
    end
end

function _do_batch_validate(learner::Learner, batch, batch_index, epoch)
    try
       #println()
       #print("_do_batch_validate ")
        _loss(xy) = learner.loss(learner.model(xy[1]),xy[2])
       #print("a")
        _cbs_before_batch_validate(learner,batch_index,epoch)
       #print("b")
        learner.batch_validate_loss = sum(_loss.(batch))
       #print("c")
        _cbs_batch_validate_loss(learner,learner.batch_validate_loss,batch_index,epoch)
    #catch CancelBatchValidateException
    #    _cbs_after_cancel_batch_validate(learner,batch_index,epoch)
    finally
        _cbs_after_batch_validate(learner,batch_index,epoch)
    end
   #println()
end

function _do_epoch_validate(learner::Learner, epoch)
    try
       #print("_do_epoch_validate ")
        _cbs_before_epoch_validate(learner,epoch)
       #print("a")
        data = learner|> data_bunch |> valid
       #print("b")
        _cbs_before_epoch_validate(learner,epoch)
       #print("c")
        for (i,batch) in enumerate(data)
            _do_batch_validate(learner,batch,i,epoch)
        end 
       #println("d")
    #catch CancelEpochValidateException
    #    _cbs_after_cancel_epoch_validate(learner,epoch)
    finally
        _cbs_after_epoch_validate(learner,epoch)
    end
end

"""
    fit(learner::Learner, n_epoch, lr=nothing, wd=nothing, cbs=nothing, reset_opt=false)

Fit learner.model for n#94epoch using cbs. Optionally reset#94opt

Uses lr and wd if they are provided, otherwise use the defaults values given by the lr and wd attributes of Learner.

All the examples use synth#94learner which is a simple Learner training a linear regression model.

```
#Training a few epochs should make the model better
learn = synth_learner(lr=1e-2)
#learn.model = learn.model.cpu() TODO
xb,yb = one_batch(learn.dls)
init_loss = loss_func(learn, learn.model(xb), yb)
fit(learn, 6)
@assert learn.loss < init_loss
```
"""
function fit!(learner::Learner, n_epoch)
    try
        _cbs_before_fit(learner,n_epoch)
        for epoch in 1:n_epoch
            try
                _cbs_before_epoch(learner,epoch)
                _do_epoch_train(learner,epoch)               
                _do_epoch_validate(learner,epoch)
            #catch CancelEpochException
            #    _cbs_after_cancel_epoch(learner,epoch)
            finally
                _cbs_after_epoch(learner,epoch)
            end
        end
    #catch CancelFitException
    #    _cbs_after_cancel_fit(learner)
    finally
        _cbs_after_fit(learner)
    end
end

function implements_learner(T::DataType)
    return hasmethod(model,(T,)) &&
        hasmethod(loss,(T,)) &&
        hasmethod(loss!,(T,Any)) &&
        hasmethod(opt,(T,)) &&
        hasmethod(opt!,(T,Any)) &&
        hasmethod(batch,(T,)) &&
        hasmethod(batch_loss,(T,)) &&
        hasmethod(batch_size,(T,)) &&
        hasmethod(add_cb!,(T,AbstractCallback))
end

#@assert implements_learner(Learner)
