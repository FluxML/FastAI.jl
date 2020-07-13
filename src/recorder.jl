#=
recorder.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the FastAI Recorder to Julia

Callback that registers statistics (lr, loss and metrics) 
during training

By default, metrics are computed on the validation set only, 
although that can be changed by adjusting train_metrics and 
valid_metrics. beta is the weight used to compute the 
exponentially weighted average of the losses (which gives 
the smooth_loss attribute to Learner).

The logger attribute of a Learner determines what happens to 
those metrics. By default, it just print them:

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/learner.py

The documentation is copied from here

https://dev.fast.ai/learner#Recorder

The main purpose of this code is to see if the team likes the method
of defining an interface and implementations in Julia
=#
"""
Callback that registers statistics (lr, loss and metrics) during training
"""
mutable struct Recorder <: AbstractCallback
    remove_on_fetch:: Bool
    # run_after TODO what does this do?
    add_time:: Bool
    train_metrics:: Bool
    valid_metrics:: Bool
    loss:: AvgLoss
    smooth_loss:: AvgSmoothLoss
    lrs:: Array
    iters:: Array
    losses:: Array
    values:: Array
end

Recorder(add_time=true, train_metrics=false, valid_metrics=true, beta=0.98) =
    Recorder(
        true,
        #TrainEvalCallback,
        add_time,
        train_metrics,
        valid_metrics,
        AvgLoss(),
        AvgSmoothLoss(beta=beta),
        [],[],[],[])

"Prepare state for training"
function begin_fit(rec::Recorder)
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

"Update all metrics and records lr and smooth loss in training"
function after_batch(rec::Recorder)
    if length(current_batch(rec)) == 0
        return
    end
    mets = if rec.training rec._train_mets else rec._valid_mets end
    for met in mets
        accumulate(met,rec.learn)
    end
    if !rec.training
        return
    end
    push!(rec.lrs.append(lr(rec.learn)))
    push!(rec.losses,value(rec.smooth_loss))
    smooth_loss!(rec.learn,value(rec.smooth_loss))
end
#=
"Set timer if `rec.add_time=true`"
function begin_epoch(rec::Recorder)
    rec.cancel_train,rec.cancel_valid = false,false
    if rc.add_time
        rec.start_epoch = time()
    end
    rec.log = L(getattr(self, 'epoch', 0))
end

    def begin_train   (self): self._train_mets[1:].map(Self.reset())
    def begin_validate(self): self._valid_mets.map(Self.reset())
    def after_train   (self): self.log += self._train_mets.map(_maybe_item)
    def after_validate(self): self.log += self._valid_mets.map(_maybe_item)
    def after_cancel_train(self):    self.cancel_train = True
    def after_cancel_validate(self): self.cancel_valid = True

    def after_epoch(self):
        "Store and log the loss/metric values"
        self.learn.final_record = self.log[1:].copy()
        self.values.append(self.learn.final_record)
        if self.add_time: self.log.append(format_time(time.time() - self.start_epoch))
        self.logger(self.log)
        self.iters.append(self.smooth_loss.count)

    @property
    def _train_mets(self):
        if getattr(self, 'cancel_train', False): return L()
        return L(self.smooth_loss) + (self.metrics if self.train_metrics else L())

    @property
    def _valid_mets(self):
        if getattr(self, 'cancel_valid', False): return L()
        return (L(self.loss) + self.metrics if self.valid_metrics else L())

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

if not hasattr(defaults, 'callbacks'): defaults.callbacks = [TrainEvalCallback, Recorder]
elif Recorder not in defaults.callbacks: defaults.callbacks.append(Recorder)
=#