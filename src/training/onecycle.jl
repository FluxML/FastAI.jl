
"""
    fitonecycle!(learner, nepochs[, lrmax])

Fit `learner` for `nepochs` using a one-cycle learning rate schedule.
The learning rate starts at `lrmax/div` for `pct_start*nepochs` epochs, rising
to `lrmax` and then goes down to `lrmax/div_final` over the remaining duration.

## Keyword arguments

- `wd = 0`: weight decay
- `pct_start = 0.25`: Percentage of time spent raising the learning rate
- `div = 25`: Starting learning rate is `lr_max/div`
- `div_final = 1e5`: Ending learning rate is `lr_max/div_final`
"""
function fitonecycle!(
        learner::Learner, nepochs::Int, maxlr=0.1;
        phases = (
            TrainingPhase() => learner.data.training,
            ValidationPhase() => learner.data.validation
        ),
        wd=0.,
        kwargs...)

    nsteps = length(phases[1][2])
    scheduler = Scheduler(LearningRate => onecycle(
        nepochs * nsteps,
        maxlr;
        kwargs...))

    wdoptim = wd > 0 ? decay_optim(learner.optimizer, wd) : learner.optimizer
    withfields(learner, optimizer=wdoptim) do
        withcallbacks(learner, scheduler) do
            for _ in 1:nepochs
                for (phase, data) in phases
                    epoch!(learner, phase, data)
                end
            end
        end
    end
end


"""
    decay_optim(optim, wd)

Add [`WeightDecay`](#) with value `wd` to optimizer `optim`.
"""
decay_optim(optim, wd) = Optimiser(WeightDecay(wd), optim)
function decay_optim(optim::Optimiser, wd)
    # change weight decay if present, else add it
    i = findfirst(o -> o isa WeightDecay, optim.os)
    if isnothing(i)
        return Optimiser(WeightDecay(wd), optim.os...)
    else
        return Optimiser(WeightDecay(wd), optim.os[1:i - 1]..., optim.os[i + 1:end]...)
    end
end


# ## Tests

@testset "decay_optim" begin
    optim = ADAM()
    @test decay_optim(optim, 0.1) isa Optimiser
    @test decay_optim(Optimiser(ADAM(), ADAM()), 0.1) isa Optimiser
    @test decay_optim(optim, 0.1).os[1] isa WeightDecay
    o = decay_optim(Optimiser(ADAM(), WeightDecay(.5)), .1)
    @test o.os[1] isa WeightDecay
    @test o.os[2] isa ADAM
end
