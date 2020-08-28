#=
recorder.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the FastAI Recorder to Julia

This code is inspired by FastAI, but differs from it in important ways

Recorders are Callbacks added to a Learner.  They keep a log of 
statistics (losses and smooth_losses) and history (opts) during training

This design is significantly different from the original 
Recorder and Metrics objects in Python

https://github.com/fastai/fastai2/blob/master/fastai2/learner.py

The documentation is copied from here

https://dev.fast.ai/learner#Recorder
=#
module Recorder
using FastAI
using FastAI: AbstractCallback,AbstractLearner
using Infiltrator
"""
    Recorder(learn::Learner; train_loss = true, train_smooth_loss = true,
                             validate_loss = true, validate_smooth_loss = true)

Container for [`Learner`](@ref) statistics (e.g. lr, loss and metrics) during training.
Statistics are indexed by name, epoch and batch.
For example to get the smoothed training loss for epoch 2, batch 3, we would call
```julia
recorder["TrainSmoothLoss", 2, 3]
```
To get the entire history of smooth training loss, one would call
```julia
recorder["TrainSmoothLoss", :, :]
```
"""
mutable struct Smoother
    alpha::Real
    val::Real
    first::Bool
end
Smoother(alpha) = Smoother(alpha, 0.0, true)

reset!(asl::Smoother) = asl.first=true
    
function (asl::Smoother)(value)
    if asl.first
        asl.first = false
        asl.val = value
    else
        asl.val = asl.alpha*asl.val+(1-asl.alpha)*value
    end
    return asl.val
end 

"""
AbstractRecorder is shared code

All Recorders have a log value, where the log supports getindex and setindex
"""
abstract type AbstractRecorder <: AbstractCallback end
FastAI.before_fit(rec::AbstractRecorder,lrn::AbstractLearner, epoch_count, batch_size) = rec.log = fill(NaN,(epoch_count,batch_size))
Base.getindex(rec::AbstractRecorder,idx...) = rec.log[idx...]

"""
    TrainLoss

Record a log of training loss
"""
mutable struct TrainLoss <: AbstractRecorder
    log::Union{Nothing,Array{Real,2}}     
end
TrainLoss()=TrainLoss(nothing)
FastAI.batch_train_loss(rec::TrainLoss,lrn::AbstractLearner, epoch, batch, loss) = rec.log[epoch,batch] = loss

"""
    ValidateLoss

Record a log of validation loss
"""
mutable struct ValidateLoss <: AbstractRecorder
    log::Union{Nothing,Array{Real,2}}
end
ValidateLoss()=ValidateLoss(nothing)
FastAI.batch_validate_loss(rec::ValidateLoss,lrn::AbstractLearner, epoch, batch, loss) = rec.log[epoch,batch] = loss

abstract type AbstractSmoothRecorder <: AbstractRecorder end

function FastAI.before_fit(rec::AbstractSmoothRecorder,lrn::AbstractLearner, epoch_count, batch_size)
     rec.log = fill(NaN,(epoch_count,batch_size))
     reset!(rec.smooth)
end

"""
    SmoothTrainLoss

Record a smoothed log of training loss
"""
mutable struct SmoothTrainLoss <: AbstractSmoothRecorder
    smooth::Smoother
    log::Union{Nothing,Array{Real,2}}     
end
SmoothTrainLoss(alpha=0.98)=SmoothTrainLoss(Smoother(alpha),nothing)
FastAI.batch_train_loss(rec::SmoothTrainLoss,lrn::AbstractLearner, epoch, batch, loss) = rec.log[epoch,batch] = rec.smooth(loss)

"""
    SmoothValidateLoss

Record a smoothed log of validation loss
"""
mutable struct SmoothValidateLoss <: AbstractSmoothRecorder
    smooth::Smoother
    log::Union{Nothing,Array{Real,2}} 
end
SmoothValidateLoss(alpha=0.98)=SmoothValidateLoss(Smoother(alpha),nothing)
FastAI.batch_validate_loss(rec::SmoothValidateLoss,lrn::AbstractLearner, epoch, batch, loss) = rec.log[epoch,batch] = rec.smooth(loss)

end
