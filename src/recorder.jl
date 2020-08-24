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

"""
    TrainLossRecorder

Record a log of training loss
"""
struct TrainLossRecorder <: AbstractCallback
    log::Array{Real,2}    
end

before_fit(lr::TrainLossRecorder,lrn::Learner, epoch_count, batch_size) = lr.log = zeros((epoch_count,batch_size))
batch_train_loss(lr::TrainLossRecorder,lrn::AbstractLearner, epoch, batch, loss) = lr.log[epoch,batch] = loss
Base.getindex(lr::TrainLossRecorder,idx...) = lr.log[idx]

"""
    ValidateLossRecorder

Record a log of validation loss
"""
struct ValidateLossRecorder <: AbstractCallback
    log::Array{Real,2}
end

before_fit(lr::TrainLossRecorder,lrn::Learner, epoch_count, batch_size) = lr.log = zeros((epoch_count,batch_size))
batch_validate_loss(lr::TrainLossRecorder,lrn::AbstractLearner, epoch, batch, loss) = lr.log[epoch,batch] = loss
Base.getindex(lr::ValidateLossRecorder,idx...) = lr.log[idx]

"""
Utility type for smoothing a series of values
"""
mutable struct Smooth
    alpha::Real
    val::Real
    first::Bool
end
Smooth(alpha) = Smooth(alpha, 0.0, true)

reset!(asl::Smooth) = asl.first=true
    
function accumulate!(asl::Smooth, value)
    if asl.first
        asl.first = false
        asl.val = value
    else
        asl.val = asl.alpha*asl.val+(1-asl.alpha)*value
    end
    return asl.val
end 

"""
    TrainSomoothLossRecorder

Record a smoothed log of training loss
"""
struct TrainSmoothLossRecorder <: AbstractCallback
    log::Array{Real,2}    
    smooth::Smoother
end

TrainSmoothLossRecorder(alpha=0.98) = TrainSmoothLossRecorder(Nothing, Smooth(alpha))

function before_fit(lr::TrainSmoothLossRecorder, lrn::Learner, epoch_count, batch_size)
    reset!(lr.smooth)
    zeros((epoch_count,batch_size))    
end

function batch_train_loss(lr::TrainSmoothLossRecorder, lrn::AbstractLearner, epoch, batch, loss)
    lr.log[epoch,batch] = accumulate!(lr.smooth, loss)
end

Base.getindex(lr::TrainSmoothLossRecorder,idx...) = lr.log[idx]

"""
    TrainSomoothLossRecorder

Record a smoothed log of validation loss
"""
struct Validate SmoothLossRecorder <: AbstractCallback
    log::Array{Real,2}    
    smooth::Smooth
end

ValidateSmoothLossRecorder(alpha=0.98) = ValidateSmoothLossRecorder(Nothing, Smooth(alpha))

function before_fit(lr::ValidateSmoothLossRecorder, lrn::Learner, epoch_count, batch_size)
    reset!(lr.smooth)
    zeros((epoch_count,batch_size))    
end

function batch_validate_loss(lr::ValidateSmoothLossRecorder, lrn::AbstractLearner, epoch, batch, loss)
    lr.log[epoch,batch] = accumulate!(lr.smooth, loss)
end

Base.getindex(lr::ValidateSmoothLossRecorder,idx...) = lr.log[idx]


#=
Recorder(add_time=true, train_metrics=false, valid_metrics=true, alpha=0.98) =
    Recorder(
        true,
        #ValidateEvalCallback,
        add_time,
        train_metrics,
        valid_metrics,
        AvgLoss(),
        AvgSmoothLoss(alpha),
        [],[],[],[])

"Prepare state for training"
function begin_fit(rec::Recorder,lrn::Learner)
    rec.lrs,rec.iters,rec.losses,rec.values = [],[],[],[]
    #=
    names = rec.metrics.attrgot('name')
    if rec.train_metrics && rec.valid_metrics
        names = L('loss') + names
        names = names.map('train_{}') + names.map('valid_{}') 
    elseif rec.valid_metrics
        names = L('train_loss', 'valid_loss') + names
    else
        names = L('train_loss') + names
    end
    
    if rec.add_time
        names.append('time')
    end
    rec.metric_names = 'epoch'+names
    =#
    reset(rec.smooth_loss)
end

begin_train(rec::Recorder) = map(reset,rec.train_metrics)
begin_validate(rec::Recorder) = map(reset,rec.validate_metrics)
after_train(rec::Recorder) = log(rec,map(maybe_item,rec.train_metrics))
after_validate(rec::Recorder) = log(rec,map(maybe_item,rec.valididate_metrics))
after_cancel_train(rec::Recorder) = rec.cancel_train = true
after_cancel_validate(rec::Recorder) = rec.cancel_valid = true

function train_metrics(rec::Recorder)
    if rec.cancel_train
        return []
    elseif rec.train_metrics
        return [rec.smooth_loss] + rec.metrics
    else
        return [rec.smooth_loss]
    end
end

function valid_metrics(rec::Recorder)
    if rec.cancel_valid
        return []
    elseif rec.valid_metrics
        return [rec.loss] + rec.metrics
    else
        return [rec.loss]
    end
end

"Update all metrics and records lr and smooth loss in training"
function after_batch(rec::Recorder)
    if batch_size(rec) > 0
        mets = if rec.is_training train_metics(rec) else valid_metrics(rec) end
        for met in mets
            accumulate(met,rec.learn)
        end
        if rec.is_training     
            push!(rec.lrs.append(lr(rec.learn)))
            push!(rec.losses,value(rec.smooth_loss))
            smooth_loss!(rec.learn,value(rec.smooth_loss))
        end
    end
end

"Set timer if `rec.add_time=true`"
function begin_epoch(rec::Recorder)
    rec.cancel_train,rec.cancel_valid = false,false
    if rc.add_time
        rec.start_epoch = time()
    end
    rec.log = L(getattr(self, 'epoch', 0))
end

"Store and log the loss/metric values"
function after_epoch(rec::Recorder)
    rec.learn.final_record = self.log[1:].copy()
    push!(rec.values,rec.learn.final_record)
    if rec.add_time
        push!(rec.log,format_time(time.time() - rec.start_epoch))
        logger(rec,rec.log)
        push!(rec.iters,count(rec.smooth_loss))
    end
end

    def plot_loss(self, skip_start=5, with_valid=True):
        plt.plot(list(range(skip_start, len(self.losses))), self.losses[skip_start:], label='train')
        if with_valid:
            idx = (np.array(self.iters)<skip_start).sum()
            plt.plot(self.iters[idx:], L(self.values[idx:]).itemgot(1), label='valid')
            plt.legend()

# Cell
add_docs(Recorder,
         begin_train = "Reset loss and metrics state",
         after_train = "Log loss and metric values on the training set (if `self.training_metrics=True`)",
         begin_validate = "Reset loss and metrics state",
         after_validate = "Log loss and metric values on the validation set",
         after_cancel_train = "Ignore training metrics for this epoch",
         after_cancel_validate = "Ignore validation metrics for this epoch",
         plot_loss = "Plot the losses from `skip_start` and onward")

if not hasattr(defaults, 'callbacks'): defaults.callbacks = [ValidateEvalCallback, Recorder]
elif Recorder not in defaults.callbacks: defaults.callbacks.append(Recorder)
=#
