#=
recorder.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the FastAI Recorder to Julia

This code is inspired by FastAI, but differs from it in important ways

Recorder is a container for Learner statistics (losses and smooth_losses) and history (opts) 
during training

By default, metrics are computed on the validation set only, 
although that can be changed by adjusting train_metrics and 
valid_metrics. alpha is the weight used to compute the 
exponentially weighted average of the losses (which gives 
the smooth_loss attribute).

The logger attribute of a Learner determines what happens to 
those metrics. By default, it just print them:

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/learner.py

The documentation is copied from here

https://dev.fast.ai/learner#Recorder
=#
"""
Container for Learner statistics (e.g. lr, loss and metrics) during training

Statistics are indexed by name, epoch and batch.  For example to get the smoothed training loss for epoch 2, batch 3, one would call

recorder["TrainSmoothLoss",2,3]

To get the entire history of smooth training loss, one would call

recorder["TrainSmoothLoss",:,:]

To add a new history to record

add!(recorder,"MyCustomMetric")
"""
struct Recorder
    learn::Learner
    logs::Dict{String,Array{Any,2}}
end

function Recorder(learn::Learner; train_loss=true, train_smooth_loss=true, validate_loss=true, validate_smooth_loss=true)
    rec = Recorder(learn,Dict{String,Array}())
    if train_loss
        add_cb!(learn,TrainLossRecorder(rec))
    end
    if train_smooth_loss
        add_cb!(learn,TrainSmoothLossRecorder(rec))
    end
    if validate_loss
        add_cb!(learn,ValidateLossRecorder(rec))
    end
    if validate_smooth_loss
        add_cb!(learn,ValidateSmoothLossRecorder(rec))
    end
    return rec
end

add!(rec::Recorder,name, epoch_count, batch_size) = rec.logs[name]=fill(nothing,epoch_count,batch_size)

function log!(rec::Recorder,name::String, epoch::Int, batch::Int, value)
#    println("$name,$epoch,$batch,$value")
    rec.logs[name][epoch,batch]=value
end

Base.getindex(rec::Recorder,idx...) = rec.logs[idx[1]][idx[2],idx[3]]
Base.setindex!(rec::Recorder,value,idx...) = rec.logs[idx[1]][idx[2],idx[3]] = value

struct TrainLossRecorder <: AbstractCallback
    rec::Recorder
end

before_fit(lr::TrainLossRecorder,lrn::Learner, epoch_count, batch_size) = add!(lr.rec,"TrainLoss", epoch_count, batch_size)
batch_train_loss(lr::TrainLossRecorder,lrn::AbstractLearner, epoch, batch, loss) = log!(lr.rec,"TrainLoss",epoch,batch,loss)

struct ValidateLossRecorder <: AbstractCallback
    rec::Recorder
end

before_fit(lr::ValidateLossRecorder,lrn::Learner, epoch_count, batch_size) = add!(lr.rec,"ValidateLoss", epoch_count, batch_size)
batch_train_loss(lr::ValidateLossRecorder,lrn::AbstractLearner, epoch, batch, loss) = log!(lr.rec,"ValidateLoss",epoch,batch,loss)

struct TrainSmoothLossRecorder <: AbstractCallback
    rec::Recorder
    smooth::SmoothMetric
end

TrainSmoothLossRecorder(rec) = TrainSmoothLossRecorder(rec,SmoothMetric())

function before_fit(lr::TrainSmoothLossRecorder,lrn::Learner, epoch_count, batch_size)
    reset!(lr.smooth)
    add!(lr.rec,"TrainSmoothLoss", epoch_count, batch_size)
end

function batch_train_loss(lr::TrainSmoothLossRecorder,lrn::AbstractLearner, epoch, batch, loss)
    accumulate!(lr.smooth,loss)
    log!(lr.rec,"TrainSmoothLoss",epoch,batch,value(lr.smooth))
end

struct ValidateSmoothLossRecorder <: AbstractCallback
    rec::Recorder
    smooth::SmoothMetric
end

ValidateSmoothLossRecorder(rec) = ValidateSmoothLossRecorder(rec,SmoothMetric())

function before_fit(lr::ValidateSmoothLossRecorder,lrn::Learner, epoch_count, batch_size)
    reset!(lr.smooth)
    add!(lr.rec,"ValidateSmoothLoss", epoch_count, batch_size)
end

function batch_validate_loss(lr::ValidateSmoothLossRecorder,lrn::AbstractLearner, epoch, batch, loss)
    accumulate!(lr.smooth,loss)
    log!(lr.rec,"ValidateSmoothLoss", epoch,batch,value(lr.smooth))
end


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
