
"""
    setschedules!(learner, schedules...)

Set `schedules` on `learner`'s `Scheduler` callback so that training resumes
from there.

If `learner` does not have a `Scheduler` callback yet, adds it.

```julia
learner = ...
fit!(learner, 1)
setschedules!(learner, onecycle(1, 0.01))
fit!(learner, 1)
```
"""
function setschedules!(learner, phase, schedules::Vararg{Pair})
    # offset by already trained epochs
    if haskey(learner.cbstate, :history)
        offset = length(learner.data.training) * learner.cbstate.history[phase].epochs
    else
        offset = 0
    end
    schedules = Tuple(HP => schedule + offset for (HP, schedule) in schedules)
    scheduler = Scheduler(schedules...)
    oldscheduler = FluxTraining.replacecallback!(learner, scheduler)
    return oldscheduler
end


"""
    freeze(model, indices)

Freeze all parameters in `model`, except those in `model[indices]`.
"""
freeze(model, indices::AbstractVector) = FrozenModel(model, model -> model[indices])


struct FrozenModel
    model
    fn
end

Flux.params(model::FrozenModel) = params(model.fn(model.model))

(m::FrozenModel)(x) = m.model(x)
