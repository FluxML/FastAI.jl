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
    lr::Real
    loss_func


    wd::Int
    n_epoch::Int

    smooth_loss
    
    # current batch
    epoch
    loss
    dl
    pb
    xb
    yb
end

Learner(db, model; opt=Flux.ADAM(), lr=0.01, loss_func=Flux.mse) = Learner([],db,model,opt,lr,loss_func, 0,0,nothing,nothing,nothing,nothing,[],[],[])

model(l::Learner) = l.model
lr(l::Learner) = l.lr
loss_func(l::Learner) = l.loss_func
loss(l::Learner) = l.loss
smooth_loss(l::Learner) = l.smooth_loss
smooth_loss!(l::Learner,sl::Real) = l.smooth_loss=sl
pb(l::Learner) = l.pb
xb(l::Learner) = l.xb
yb(l::Learner) = l.yb
batch_size(l::Learner) = length(l.yb)
data_bunch(l::Learner) = l.db
loss(l::Learner,xb,yb) = l.loss


"""
add_cb(learner::Learner,cb::AbstractCallback cb)

Add a new Callback [AbstractCallback](@ref) to this Learner [Learner](@ref)
"""
add_cb!(learner::Learner,cb::AbstractCallback) = push!(learner.cbs,cb)

# pass event to all callbacks
_cbs_begin_fit(learner::Learner) =  for cb in learner.cbs begin_fit(cb,learner) end
_cbs_after_fit(learner::Learner) =  for cb in learner.cbs after_fit(cb,learner) end
_cbs_after_cancel_fit(learner::Learner) =  for cb in learner.cbs after_cancel_fit(cb,learner) end
_cbs_begin_train(learner::Learner) =  for cb in learner.cbs begin_train(cb,learner) end
_cbs_after_train(learner::Learner) =  for cb in learner.cbs after_train(cb,learner) end
_cbs_after_cancel_train(learner::Learner) =  for cb in learner.cbs after_cancel_train(cb,learner) end
_cbs_begin_epoch(learner::Learner) =  for cb in learner.cbs begin_epoch(cb,learner) end
_cbs_after_epoch(learner::Learner) =  for cb in learner.cbs after_epoch(cb,learner) end
_cbs_after_cancel_epoch(learner::Learner) =  for cb in learner.cbs after_cancel_epoch(cb,learner) end
_cbs_begin_batch(learner::Learner) =  for cb in learner.cbs begin_batch(cb,learner) end
_cbs_after_batch(learner::Learner) =  for cb in learner.cbs after_batch(cb,learner) end
_cbs_begin_validate(learner::Learner) =  for cb in learner.cbs begin_validate(cb,learner) end
_cbs_after_validate(learner::Learner) =  for cb in learner.cbs after_validate(cb,learner) end
_cbs_after_pred(learner::Learner) =  for cb in learner.cbs after_pred(cb,learner) end
_cbs_after_loss(learner::Learner) =  for cb in learner.cbs after_loss(cb,learner) end
_cbs_after_backward(learner::Learner) =  for cb in learner.cbs after_backward(cb,learner) end
_cbs_after_step(learner::Learner) =  for cb in learner.cbs after_step(cb,learner) end
_cbs_after_cancel_batch(learner::Learner) =  for cb in learner.cbs after_cancel_batch(cb,learner) end

function _do_batch_fit(learner::Learner, batch, ps)

    _loss(xy) = learner.loss_func(learner.model(xy[1]),xy[2])
    _cbs_begin_batch(learner)
    gs = gradient(ps) do
        sum(_loss.(batch))
    end
    update!(learner.opt, ps, gs)
    _cbs_after_batch(learner)
end

function _do_epoch_train(learner::Learner)
    try
        learner.dl = learner|> data_bunch |> train
        ps = params(learner.model)
        _cbs_begin_train(learner)
        for batch in learner.dl
            _do_batch_fit(learner,batch,ps)
        end 
    catch CancelTrainException
        _cbs_after_cancel_train(learner)
    finally
        _cbs_after_train(learner)
    end
end

function _do_epoch_validate(learner::Learner, ds_idx=1, dl=nothing)
    dl = isnothing(dl) ? learner.db[ds_idx] : dl
    try
        learner.dl = dl
        _cbs_begin_validate(learner)
        # with torch.no_grad(): TODO
        all_batches(learner)
    catch CancelValidException
        _cbs_after_cancel_validate(learner)
    finally
        _cbs_after_validate(learner)
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
function fit!(learner::Learner, n_epoch, lr=nothing, wd=nothing, cbs=nothing, reset_opt=false)
    #=
    if reset_opt || isnothing(learner.opt)
        create_opt(learner)
    end
    wd = isnothing(wd) ? learner.wd : wd
    if !isnothing(wd)
        set_hypers(learner.opt,wd=wd)
    end
    set_hypers(learner.opt, lr= isnothing(lr) ? learner.lr : lr)
    =#
    try
        learner.n_epoch = n_epoch
        learner.loss = 0.0
        _cbs_begin_fit(learner)
        for epoch in 1:n_epoch
            try
                learner.epoch=epoch
                _cbs_begin_epoch(learner)
                _do_epoch_train(learner)
                _do_epoch_validate(learner)
            catch CancelEpochException
                _cbs_after_cancel_epoch(learner)
            finally
                _cbs_after_epoch(learner)
            end
        end
    catch CancelFitException
        _cbs_after_cancel_fit(learner)
    finally
        _cbs_after_fit(learner)
        learner.dl,learner.xb,learner.yb,learner.pb,learner.loss = nothing,(nothing,),(nothing,),nothing,nothing
    end
end

function implements_learner(T::DataType)
    return hasmethod(lr,(T,)) &&
        hasmethod(loss,(T,)) &&
        hasmethod(smooth_loss,(T,)) &&
        hasmethod(smooth_loss!,(T,Real)) &&
        hasmethod(batch_size,(T,)) &&
        hasmethod(xb,(T,)) &&
        hasmethod(yb,(T,)) &&
        hasmethod(pb,(T,)) &&
        hasmethod(add_cb!,(T,AbstractCallback))
end

#@assert implements_learner(Learner)