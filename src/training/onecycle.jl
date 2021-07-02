
"""
    fitonecycle!(learner, nepochs[, lrmax])

Fit `learner` for `nepochs` using a one-cycle learning rate schedule.
"""
function fitonecycle!(
        learner, nepochs, maxlr = 0.1;
        dataiters = (learner.data.training, learner.data.validation),
        kwargs...)
    nsteps = length(learner.data.training)
    scheduler = Scheduler(LearningRate => onecycle(
        nepochs * nsteps,
        maxlr;
        kwargs...))
    withcallbacks(learner, scheduler) do
        fit!(learner, nepochs, dataiters)
    end
end
"""
function fitonecycle!(
        learner,
        nepochs,
        lrmax = 0.01;
        div = 25,
        divfinal = 1e5,
        pct_start = 0.25)





    nsteps = length(learner.data.training)
    schedules = (
        LearningRate => onecycle(nepochs * nsteps, lrmax; div = div, divfinal = divfinal, pct_start = pct_start),
    )
    oldscheduler = setschedules!(learner, TrainingPhase(), schedules...)

    try
        fit!(learner, nepochs)
    catch e
        rethrow(e)
    finally
        # reset to previous Scheduler
        if !isnothing(oldscheduler)
            FluxTraining.replacecallback!(learner, oldscheduler)
        end
    end
end
"""
