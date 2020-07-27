#=
callback.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the FastAI V2 Callback API to Julia

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/callback/core.py

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/callback.core.html

The main purpose of this code is to see if the team likes the method
of defining an interface and implementations in Julia
=#

"called before train and validate, ideal for initial setup."
function before_fit(cb::AbstractCallback,lrn::AbstractLearner, n_epoch) println("Before Fit") end
"called at the after train and validate, for final clean-up."
function after_fit(cb::AbstractCallback,lrn::AbstractLearner) println("After Fit") end
"called after cancelling train and validate, for final clean-up."
function after_cancel_fit(cb::AbstractCallback,lrn::AbstractLearner) println("After Cancel Fit") end

"called before each epoch"
function before_epoch(cb::AbstractCallback,lrn::AbstractLearner, epoch) println("Before Epoch") end
"called after each epoch"
function after_epoch(cb::AbstractCallback,lrn::AbstractLearner, epoch) println("After Epoch") end
"called after cancelling epoch"
function after_cancel_epoch(cb::AbstractCallback,lrn::AbstractLearner, epoch) println("After Cancel Epoch") end

"called at the beginning of the training part of an epoch."
function before_epoch_train(cb::AbstractCallback,lrn::AbstractLearner,epoch) println("\nBefore Epoch Train") end
"called at the end of the training phase of an epoch."
function after_epoch_train(cb::AbstractCallback,lrn::AbstractLearner,epoch) println("\nAfter Epoch Train") end
"called after cancelling the training phase of an epoch."
function after_cancel_epoch_train(cb::AbstractCallback,lrn::AbstractLearner,epoch) println("\nAfter Cancel Epoch Train") end

"called at the beginning of each epoch, useful for any behavior you need to reset at each epoch."
function before_epoch_validate(cb::AbstractCallback,lrn::AbstractLearner,epoch) println("\nBefore Epoch Validate") end
"called at the end of an epoch, for any clean-up before the next one."
function after_epoch_validate(cb::AbstractCallback,lrn::AbstractLearner,epoch) println("\nAfter Epoch Validate") end
"called after cancelling an epoch, for any clean-up before the next one."
function after_cancel_epoch_validate(cb::AbstractCallback,lrn::AbstractLearner,epoch) println("\nAfter Cancel Epoch Validate") end

"called at the beginning of each batch, just after drawing said batch. It can be used to do any setup necessary for the batch (like hyper-parameter scheduling) or to change the input/target before it goes in the model (change of the input with techniques like mixup for instance)."
function before_batch_train(cb::AbstractCallback,lrn::AbstractLearner,batch_index, epoch) print("Before Batch Train,") end
"called at the end of a batch, for any clean-up before the next one."
function after_batch_train(cb::AbstractCallback,lrn::AbstractLearner,batch_index, epoch) print("After Batch Train,")  end
"called after cancelling an epoch, for any clean-up before the next one."
function after_cancel_batch_train(cb::AbstractCallback,lrn::AbstractLearner,batch_index, epoch) print("After Cancel Batch Train")  end

"called at the beginning of each batch validate, just after drawing said batch. It can be used to do any setup necessary for the batch (like hyper-parameter scheduling) or to change the input/target before it goes in the model (change of the input with techniques like mixup for instance)."
function before_batch_validate(cb::AbstractCallback,lrn::AbstractLearner,batch_index, epoch) print("Before Batch Validate,")  end
"called at the end of a batch validate, for any clean-up before the next one."
function after_batch_validate(cb::AbstractCallback,lrn::AbstractLearner,batch_index, epoch) print("After Batch Validate") end
"called to report the loss of a batch validate"
function batch_validate_loss(cb::AbstractCallback,lrn::AbstractLearner,loss,batch_index, epoch) println("\nBatch Validate Loss = $(loss)") end
"called after cancelling a batch validate, for any clean-up before the next one."
function after_cancel_batch_validate(cb::AbstractCallback,lrn::AbstractLearner,batch_index, epoch) println("After Cancel Batch Validate") end

struct DummyCallback <: AbstractCallback end

"`Callback` that tracks the epoch and batch and calculates progress (fraction done)"
mutable struct ProgressCallback <: AbstractCallback
    epoch::Int
    batch::Int
    n_batch::Int
    n_epoch::Int
end

ProgressCallback() = ProgressCallback(0,0,0,0)

function before_fit(tecb::ProgressCallback,learn::AbstractLearner, n_epoch)
    tecb.n_epoch = n_epoch
    tecb.n_batch = 0
    tecb.epoch = 0
    tecb.batch = 0
end

function after_epoch_train(tecb::ProgressCallback,lrn::AbstractLearner, epoch)
    tecb.n_batch = batch
    tecb.epoch = epoch
end

function after_batch_train(tecb::ProgressCallback,lrn::AbstractLearner, epoch, batch)
    tecb.batch = batch
end

function after_batch_validate_(tecb::ProgressCallback,lrn::AbstractLearner, loss, epoch, batch)
    println("Loss=$(loss) Amount trained = $(progress(tecb))")
end

function progress(tecb::ProgressCallback) 
    num = ((tecb.n_epoch-1)*tecb.n_batch + tecb.batch)
    deom = float(tecb.n_epoch*tecb.n_batch)
    return denom > 0.0 ? num / denom : 0.0
end

