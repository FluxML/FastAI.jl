
"""
    finetune!(learner, nepochs[, base_lr = 0.002; kwargs...])

Behaves as [`fastai.Learner.fine_tune`](https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py#L151)
"""
function finetune!(
        learner,
        nepochs,
        base_lr = 0.002;
        freezeepochs = 1,
        trainlayers = [2,],
        lr_mult = 100,
        div = 5,
        pct_start = 0.3,
        kwargs...)
    FluxTraining.initlearner!(learner, [TrainingPhase()])
    # freeze backbone and train head
    withfields(learner, model = (FluxTraining.model!, freeze(learner.model, trainlayers))) do
        fitonecycle!(learner, freezeepochs, base_lr, pct_start=0.99; kwargs...)
    end

    base_lr /= 2
    # TODO: use discriminative learning rates
    fitonecycle!(
        learner, nepochs, base_lr;
        div = div, pct_start = pct_start, kwargs...)
    return learner
end
