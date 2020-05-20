#=
Callback.jl:

Author: Peter Wolf (opus111@gmail.com)

A first cut at a port of the FastAI V2 Callback API to Julia

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/callback/core.py

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/callback.core.html

The main purpose of this code is to see if the team likes the method
of defining an interface and implementations in Julia
=#

module cb

"""
Basic class handling tweaks of the training loop by changing a <a href="/13a_learner#AbstractLearner"><code>AbstractLearner</code></a> in various events

The training loop is defined in <a href="/13a_learner#AbstractLearner"><code>AbstractLearner</code></a> a bit below and consists in a minimal set of instructions: looping through the data we:

- compute the output of the model from the input
- calculate a loss between this output and the desired target
- compute the gradients of this loss with respect to all the model parameters
- update the parameters accordingly
- zero all the gradients

Any tweak of this training loop is defined in a <a href="/callback.core#Callback"><code>Callback</code></a> to avoid over-complicating the code of the training loop, and to make it easy to mix and match different techniques (since they'll be defined in different callbacks). A callback can implement actions on the following events:

- begin_fit
- after_fit
- begin_train
- after_train
- begin_epoch
- after_epoch
- begin_batch
- after_batch
- begin_validate
- after_validate
- after_pred
- after_loss
- after_backward
- after_step
- after_cancel_batch
- after_batch

By default handling of these events do nothing.  Special behavior is implemented by overriding these methods

"""
abstract type AbstractCallback end

"called before doing anything, ideal for initial setup."
function begin_fit(cb::AbstractCallback,lrn::AbstractLearner) end
"called at the end of training, for final clean-up."
function after_fit(cb::AbstractCallback,lrn::AbstractLearner) end

"called at the beginning of the training part of an epoch."
function begin_train(cb::AbstractCallback,lrn::AbstractLearner) end
"called at the end of the training phase of an epoch."
function after_train(cb::AbstractCallback,lrn::AbstractLearner) end

"called at the beginning of each epoch, useful for any behavior you need to reset at each epoch."
function begin_epoch(cb::AbstractCallback,lrn::AbstractLearner) end
"called at the end of an epoch, for any clean-up before the next one."
function after_epoch(cb::AbstractCallback,lrn::AbstractLearner) end

"called at the beginning of each batch, just after drawing said batch. It can be used to do any setup necessary for the batch (like hyper-parameter scheduling) or to change the input/target before it goes in the model (change of the input with techniques like mixup for instance)."
function begin_batch(cb::AbstractCallback,lrn::AbstractLearner) end
"called at the end of a batch, for any clean-up before the next one."
function after_batch(cb::AbstractCallback,lrn::AbstractLearner) end

"called at the beginning of the validation phase of an epoch, useful for any setup needed specifically for validation."
function begin_validate(cb::AbstractCallback,lrn::AbstractLearner) end
"called at the end of the validation part of an epoch."
function after_validate(cb::AbstractCallback,lrn::AbstractLearner) end

"called after computing the output of the model on the batch. It can be used to change that output before it's fed to the loss."
function after_pred(cb::AbstractCallback,lrn::AbstractLearner) end

"called after the loss has been computed, but before the backward pass. It can be used to add any penalty to the loss (AR or TAR in RNN training for instance).<"
function after_loss(cb::AbstractCallback,lrn::AbstractLearner) end

"called after the backward pass, but before the update of the parameters. It can be used to do any change to the gradients before said update (gradient clipping for instance)."
function after_backward(cb::AbstractCallback,lrn::AbstractLearner) end

"called after the step and before the gradients are zeroed."
function after_step(cb::AbstractCallback,lrn::AbstractLearner) end

"called after a batch has been cancelled"
function after_cancel_batch(cb::AbstractCallback,lrn::AbstractLearner) end

"`Callback` that tracks the number of iterations done and properly sets training/eval mode"
struct TrainEvalCallback <: Callback
    run_valid::Bool = false
    iter::Int = 0
    epoch::Int = 0
    n_iter::Int = 0
    n_epoch::Int = 0
end

"Set the iter and epoch counters to 0, put the model and the right device"
function begin_fit(tecb::TrainEvalCallback,learn::AbstractLearner)
    learn.train_iter,learn.pct_train = 0,0.
    # self.model.to(self.dls.device) TODO how should be do this
end

"Update the iter counter (in training mode)"
function after_batch(tecb::TrainEvalCallback,lrn::AbstractLearner)
    learn.pct_train += 1./(tecb.n_iter*tecb.n_epoch)
    learn.train_iter += 1
end

"Set the model in training mode"
function begin_train(tecb::TrainEvalCallback,lrn::AbstractLearner)
    learn.pct_train=self.epoch/self.n_epoch
    # self.model.train() TODO
    learn.training=true
end

"Set the model in validation mode"
function begin_validate(tecb::TrainEvalCallback,lrn::AbstractLearner)
    # learn.model.eval() TODO
    learn.training=false
end

end
