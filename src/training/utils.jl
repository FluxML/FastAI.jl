
"""
    setschedules!(learner, phase, schedules...)

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
    withfields(f, x; kwargs...)

Replace fields on `x` with given keyword arguments, run `f` and then
restore the fields. `x` needs to be a `mutable struct`.

Every keyword argument is a mapping `(field, value)` or `(field, (setfn!, value))`.
`setfn!(x, val)` will be used to set the field; if as in the first case none
is given `setfield!` is used.
"""
function withfields(f, x; kwargs...)
    values = Dict{Symbol, Any}()
    try
        for (field, value) in kwargs
            if value isa Tuple
                setfn!, val = value
            else
                setfn! = (obj, val) -> setfield!(obj, field, val)
                val = value
            end
            values[field] = getfield(x, field)
            setfn!(x, val)
        end
        f()
    catch e
        rethrow(e)
    finally
        for (field, value) in kwargs
            if value isa Tuple
                setfn!, val = value
            else
                setfn! = (obj, val) -> setfield!(obj, field, val)
                val = value
            end
            setfn!(x, values[field])
        end
    end
end


"""
    withcallbacks(f, learner, callbacks...)

Run `f` with `callbacks` on `learner`. Existing callbacks on `learner` of
the same type as in `callbacks` are swapped during the execution of `f`.
"""
function withcallbacks(f, learner, callbacks...)
    origcallbacks = [FluxTraining.replacecallback!(learner, cb) for cb in callbacks]
    try
        f()
    catch e
        rethrow(e)
    finally
        for (i, cb) in enumerate(origcallbacks)
            if isnothing(cb)
                FluxTraining.removecallback!(learner, typeof(callbacks[i]))
            else
                FluxTraining.replacecallback!(learner, cb)
            end
        end
    end
end


"""
    makebatch(task, data, [idxs; context]) -> (xs, ys)

Create a batch of encoded data by loading `idxs` from data container `data`.
Useful for inspection and as input to [`plotbatch`](#). Samples are encoded
in `context` which defaults to `Training`.
"""
function makebatch(task::LearningTask, data, idxs = 1:8; context = Training())
    xys = [deepcopy(encodesample(task, context, getobs(data, i))) for i in idxs]
    return DataLoaders.collate(xys)
end
