#=
Learner.jl:

Author: Peter Wolf (opus111@gmail.com)

A first cut at a port of the FastAI V2 Learner API to Julia

Basic class for handling the training loop

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/learner.py

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/learner.html

The main purpose of this code is to see if the team likes the method
of defining an interface and implementations in Julia
=#

"""
add_cb(learner::Learner,cb::AbstractCallback cb)

Add a new Callback [AbstractCallback](@ref) to this Learner [Learner](@ref)
"""
add_cb(learner::Learner,cb::AbstractCallback) = push!(learner.cbs,cb)

# pass event to all callbacks
_cbs_begin_fit(learner::Learner) =  for c in learner.cbs cb.begin_fit(c,learner) end
_cbs_after_fit(learner::Learner) =  for c in learner.cbs cb.after_fit(c,learner) end
_cbs_begin_train(learner::Learner) =  for c in learner.cbs cb.begin_train(c,learner) end
_cbs_after_train(learner::Learner) =  for c in learner.cbs cb.after_train(c,learner) end
_cbs_begin_epoch(learner::Learner) =  for c in learner.cbs cb.begin_epoch(c,learner) end
_cbs_after_epoch(learner::Learner) =  for c in learner.cbs cb.after_epoch(c,learner) end
_cbs_begin_batch(learner::Learner) =  for c in learner.cbs cb.begin_batch(c,learner) end
_cbs_after_batch(learner::Learner) =  for c in learner.cbs cb.after_batch(c,learner) end
_cbs_begin_validate(learner::Learner) =  for c in learner.cbs cb.begin_validate(c,learner) end
_cbs_after_validate(learner::Learner) =  for c in learner.cbs cb.after_validate(c,learner) end
_cbs_after_pred(learner::Learner) =  for c in learner.cbs cb.after_pred(c,learner) end
_cbs_after_loss(learner::Learner) =  for c in learner.cbs cb.after_loss(c,learner) end
_cbs_after_backward(learner::Learner) =  for c in learner.cbs cb.after_backward(c,learner) end
_cbs_after_step(learner::Learner) =  for c in learner.cbs cb.after_step(c,learner) end
_cbs_after_cancel_batch(learner::Learner) =  for c in learner.cbs cb.after_cancel_batch(c,learner) end
_cbs_after_batch(learner::Learner) =  for c in learner.cbs cb.after_batch(c,learner) end

function _do_begin_fit(learner::Learner, n_epoch)
    learner.n_epoch = n_epoch
    learner.loss = 0.0
    _cbs_begin_fit(learner)
end

function _do_epoch_train(learner::Learner)
    try
        learner.dl = learner.dls.train
        _cbs_begin_train(learner)
        all_batches(learner)
    catch CancelTrainException
        _cbs_after_cancel_train(learner)
    finally
        _cbs_after_train(learner)
    end
end

function _do_epoch_validate(learner::Learner, ds_idx=1, dl=nothing)
    dl = isnothing(dl) ? learner.dls[ds_idx] : dl
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

function _end_cleanup(learner::Learner)
    learner.dl,learner.xb,learner.yb,learner.pred,learner.loss = nothing,(nothing,),(nothing,),nothing,nothing
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
function fit(learner::Learner, n_epoch, lr=nothing, wd=nothing, cbs=nothing, reset_opt=false)
    if reset_opt || isnothing(learner.opt)
        create_opt(learner)
    end
    wd = isnothing(wd) ? learner.wd : wd
    if !isnothing(wd)
        set_hypers(learner.opt,wd=wd)
    end
    set_hypers(learner.opt, lr= isnothing(lr) ? learner.lr : lr)

    try
        _do_begin_fit(learner,n_epoch)
        for epoch in range(n_epoch)
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
        _end_cleanup(learner)
    end
end
