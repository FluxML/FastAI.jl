
"""
    finetune!(learner, nepochs[, base_lr = 0.002; kwargs...])

Behaves like the fastai implementation
[`fastai.Learner.fine_tune`](https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py#L151).

## Keyword arguments

- `freezeepochs = 1`: Number of epochs to train with the backbone completely frozen.
- `grouper = FastAI.defaultgrouper(learner.model)`: [`ParamGrouper`](#) which assigns
    groups `1` (backbone) or `2` (head) for every parameter in `learner.model`. The
    default expects `learner.model` to be a `Chain(backbone, head)`.
- `backbone_factor = 0.1`: Factor by which updates to backbone model are discounted
    during the second phase of training.

Any additional keyword arguments are passed to [`fitonecycle!`](#).
"""
function finetune!(
        learner,
        nepochs,
        base_lr=0.002;
        freezeepochs=1,
        grouper=defaultgrouper(learner.model),
        backbone_factor=0.1,
        div=5,
        kwargs...)

    # Freeze backbone and train head
    foptim = frozen_optimizer(learner.optimizer, grouper, learner.model)
    withfields(learner, optimizer=foptim) do
        fitonecycle!(learner, freezeepochs, base_lr, pct_start=0.99; kwargs...)
    end

    # Use discriminative learning rates on backbone and train some more
    doptim = discrlr_optimizer(learner.optimizer, grouper, learner.model, backbone_factor)
    withfields(learner, optimizer=doptim) do
        fitonecycle!(
            learner, nepochs, base_lr / 2;
            div=div, pct_start=pct_start, kwargs...)
    end

    return learner
end


"""
    frozen_optimizer(optim, grouper, model)

Create an optimizer that only updates parameters which [`ParamGrouper`](#)
puts into group `2`.
"""
frozen_optimizer(optim, grouper, model) = discrlr_optimizer(optim, grouper, model, 0.)


"""
    frozen_optimizer(optim, grouper, model, factor)

Create an optimizer that discounts updates parameters which [`ParamGrouper`](#)
puts into group `1` by `factor`.
"""
function discrlr_optimizer(optim, grouper, model, factor)
    paramgroups = ParamGroups(grouper, model)
    return Optimiser(
        DiscriminativeLRs(paramgroups, Dict(1 => factor, 2 => 1.)),
        optim,
    )
end


function defaultgrouper(model)
    if !(model isa Chain) && length(model) == 2
        error(
            "Cannot freeze `learner.model` automatically since it is not a `Chain`.
            Please provide a `ParamGrouper` with the `grouper` keyword argument.
            The `grouper` should assign groups `1` (backbone) and `2` (head).
            ")
    else
        return IndexGrouper([1:length(model) - 1, length(model)])
    end
end
