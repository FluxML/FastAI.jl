#=
learner.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the FastAI V2 Learner API to Julia

Methods for handling the training loop

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/learner.py

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/learner.html

=#

"""
Basic class handling tweaks of the training loop by changing a [Learner](@ref) in various events

The training loop is defined in [Learner](@ref) a bit below and consists in a minimal set of instructions: looping through the data we:

compute the output of the model from the input
calculate a loss between this output and the desired target
compute the gradients of this loss with respect to all the model parameters
update the parameters accordingly

Any tweak of this training loop is defined in a Callback to avoid over-complicating the code of the training loop, and to make it easy to mix and match different techniques (since they'll be defined in different callbacks).

A callback can implement the following methods:

before_fit
after_fit
after_cancel_fit

before_epoch
after_epoch
after_cancel_epoch

before_epoch_train
after_epoch_train
after_cancel_epoch_train

before_batch_train
batch_train_loss
after_batch_train
after_cancel_batch_train

before_epoch_validate
after_epoch_validate
after_cancel_epoch_validate

before_batch_validate
after_batch_validate
batch_validate_loss
after_cancel_batch_validate

By default handling of these events do nothing.  Special behavior is implemented by overriding these methods

"""
abstract type AbstractCallback end

struct CancelFitException <: Exception end
struct CancelEpochTrainException <: Exception end
struct CancelBatchTrainException <: Exception end
struct CancelEpochValidateException <: Exception end
struct CancelBatchValidateException <: Exception end

"""
    AbstractLearner

An `AbstractLearner` groups together a model, train and validate data,
  optimizer, loss function and callbacks.

[Callbacks](@ref) are used for every tweak of the training loop.
Callbacks receive epoch, batch and loss information which they may pass on to [Metrics](@ref).

# Require interface
- `data_bunch(l::AbstractLearner)`
- `data_bunch!(l::AbstractLearner, data_bunch)`
- `model(l::AbstractLearner)`
- `model!(l::AbstractLearner, model)`
- `loss(l::AbstractLearner)`
- `loss!(l::AbstractLearner, loss)`
- `opt(l::AbstractLearner)`
- `opt!(l::AbstractLearner, opt)`
- `add_cb!(learner::AbstractLearner, cb::AbstractCallback)`
- `cbs(learner::AbstractLearner)`
- `fit!(learner::AbstractLearner, epoch_count)`
"""
abstract type AbstractLearner end

"""
    Learner <: AbstractLearner
    Learner(data_bunch, model; opt = Flux.ADAM(), loss = Flux.mse)

A `Learner` is the standard grouping of a data bunch, model, optimizer, and loss.
"""
mutable struct Learner <: AbstractLearner
    cbs:: Array{AbstractCallback}
    db::DataBunch
    model
    opt
    loss
end
Learner(data_bunch, model; opt=Flux.ADAM(), loss=Flux.mse) = Learner([],data_bunch,model,opt,loss)

"""
    data_bunch(l::Learner)

Get the data bunch for `l`.
"""
data_bunch(l::Learner) = l.db
"""
    data_bunch!(l::Learner, data_bunch)

Set the data bunch for `l` to `data_bunch`.
"""
data_bunch!(l::Learner, data_bunch) = l.db = data_bunch

"""
    model(l::Learner)

Get the model for `l`.
"""
model(l::Learner) = l.model
"""
    model!(l::Learner, model)

Set the model for `l` to `model`.
"""
model!(l::Learner, model) = l.model = model

"""
    loss(l::Learner)

Get the loss for `l`.
"""
loss(l::Learner) = l.loss
"""
    loss!(l::Learner, loss)

Set the loss for `l` to `loss`.
"""
loss!(l::Learner,loss) = l.loss = loss

"""
    opt(l::Learner)

Get the optimizer for `l`.
"""
opt(l::Learner) = l.opt
"""
    opt!(l::Learner, opt)

Set the optimizer for `l` to `opt`.
"""
opt!(l::Learner,opt) = l.opt=opt

"""
    add_cb!(learner::Learner, cb::AbstractCallback)

Add `cb` to the list of callbacks for `learner`.
"""
add_cb!(learner::Learner,cb::AbstractCallback) = push!(learner.cbs,cb)
"""
    cbs(learner::Learner)

Get the list of callbacks for `learner`.
"""
cbs(learner::Learner) = learner.cbs

# pass event to all callbacks
_cbs_before_fit(learner::Learner, epoch_count, batch_size) =  for cb in learner.cbs before_fit(cb,learner,epoch_count,batch_size) end
_cbs_after_fit(learner::Learner) =  for cb in learner.cbs after_fit(cb,learner) end
_cbs_after_cancel_fit(learner::Learner) =  for cb in learner.cbs after_cancel_fit(cb,learner) end

_cbs_before_epoch(learner::Learner, epoch) =  for cb in learner.cbs before_epoch(cb,learner,epoch) end
_cbs_after_epoch(learner::Learner, epoch) =  for cb in learner.cbs after_epoch(cb,learner,epoch) end
_cbs_after_cancel_epoch(learner::Learner, epoch) =  for cb in learner.cbs after_cancel_epoch(cb,learner,epoch) end

_cbs_before_epoch_train(learner::Learner, epoch) =  for cb in learner.cbs before_epoch_train(cb,learner,epoch) end
_cbs_after_epoch_train(learner::Learner, epoch) =  for cb in learner.cbs after_epoch_train(cb,learner,epoch) end
_cbs_after_cancel_epoch_train(learner::Learner, epoch) =  for cb in learner.cbs after_cancel_epoch_train(cb,learner,epoch) end

_cbs_before_batch_train(learner::Learner, epoch, batch) =  for cb in learner.cbs before_batch_train(cb,learner, epoch, batch) end
_cbs_batch_train_loss(learner::Learner, epoch, batch, loss) =  for cb in learner.cbs batch_train_loss(cb,learner, epoch, batch, loss) end
_cbs_after_batch_train(learner::Learner, batch, epoch) =  for cb in learner.cbs after_batch_train(cb,learner, epoch, batch) end
_cbs_after_cancel_batch_train(learner::Learner, batch, epoch) =  for cb in learner.cbs after_cancel_batch_train(cb,learner, epoch, batch) end

_cbs_before_epoch_validate(learner::Learner, epoch) =  for cb in learner.cbs before_epoch_validate(cb,learner,epoch) end
_cbs_after_epoch_validate(learner::Learner, epoch) =  for cb in learner.cbs after_epoch_validate(cb,learner,epoch) end
_cbs_after_cancel_epoch_validate(learner::Learner, epoch) =  for cb in learner.cbs after_cancel_epoch_validate(cb,learner,epoch) end

_cbs_before_batch_validate(learner::Learner, epoch, batch) =  for cb in learner.cbs before_batch_validate(cb,learner, epoch, batch) end
_cbs_after_batch_validate(learner::Learner, epoch, batch) =  for cb in learner.cbs after_batch_validate(cb,learner, epoch, batch) end
_cbs_batch_validate_loss(learner::Learner, epoch, batch, loss) =  for cb in learner.cbs batch_validate_loss(cb,learner, epoch, batch, loss) end
_cbs_after_cancel_batch_validate(learner::Learner, epoch, batch) =  for cb in learner.cbs after_cancel_batch_validate(cb,learner, epoch, batch) end

function _do_batch_train(learner::Learner, batch_data, epoch, batch, ps)
    try
        _loss(xy) = learner.loss(learner.model(xy[1]),xy[2])
        _cbs_before_batch_train(learner,epoch, batch)
        loss, back = Zygote.pullback(ps) do
            sum(_loss.(batch_data))
        end
        _cbs_batch_train_loss(learner, epoch, batch, loss)
        gs = back(1)        
        update!(learner.opt, ps, gs)
    #catch CancelBatchTrainException
    #    _cbs_after_cancel_batch_train(learner,batch_index,epoch)
    finally
        _cbs_after_batch_train(learner,batch,epoch)
    end
end

function _do_epoch_train(learner::Learner, epoch, ps)
    try
        data = learner|> data_bunch |> train
        _cbs_before_epoch_train(learner,epoch)
        for (batch,batch_data) in enumerate(data)
            _do_batch_train(learner,batch_data,epoch,batch,ps)
        end 
    #catch CancelEpochTrainException
    #    _cbs_after_cancel_epoch_train(learner,epoch)
    finally
        _cbs_after_epoch_train(learner,epoch)
    end
end

function _do_batch_validate(learner::Learner, batch_data, epoch, batch)
    try
        _loss(xy) = learner.loss(learner.model(xy[1]),xy[2])
        _cbs_before_batch_validate(learner,epoch,batch)
        loss = sum(_loss.(batch_data))
        _cbs_batch_validate_loss(learner,epoch,batch,loss)
    #catch CancelBatchValidateException
    #    _cbs_after_cancel_batch_validate(learner,epoch,batch)
    finally
        _cbs_after_batch_validate(learner,epoch,batch)
    end
end

function _do_epoch_validate(learner::Learner, epoch)
    try
        _cbs_before_epoch_validate(learner,epoch)
        data = learner|> data_bunch |> valid
        _cbs_before_epoch_validate(learner,epoch)
        for (batch,batch_data) in enumerate(data)
            _do_batch_validate(learner,batch_data,epoch,batch)
        end 
    #catch CancelEpochValidateException
    #    _cbs_after_cancel_epoch_validate(learner,epoch)
    finally
        _cbs_after_epoch_validate(learner,epoch)
    end
end

"""
    fit(learner::Learner, epoch_count)

Fit [`model(learner)`](@ref model(::Learner)) for `epoch_count` epochs invoking callbacks for `learner`.
"""
function fit!(learner::Learner, epoch_count)
    try
        batch_size = train(data_bunch(learner)).batchsize
        ps = params(learner.model)
        #@infiltrate
        _cbs_before_fit(learner,epoch_count,batch_size)
        for epoch in 1:epoch_count
            try
                _cbs_before_epoch(learner,epoch)
                _do_epoch_train(learner,epoch,ps)               
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

"""
    implements_learner(T::DataType)

Test if a type implements the [`AbstractLearner`](@ref) interface.
"""
function implements_learner(T::DataType)
    return hasmethod(model,(T,)) &&
        hasmethod(model!,(T,Any)) &&
        hasmethod(loss,(T,)) &&
        hasmethod(loss!,(T,Any)) &&
        hasmethod(opt,(T,)) &&
        hasmethod(opt!,(T,Any)) &&
        hasmethod(cbs,(T,)) &&
        hasmethod(add_cb!,(T,AbstractCallback))
end

@assert implements_learner(Learner)
