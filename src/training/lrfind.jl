
struct LRFinderResult
    losses
    lrs
    estimators
    estimates
end


"""
    lrfind(learner[, dataiter; kwargs...]) -> LRFinderResult

Run the learning rate finder. Exponentially increases the learning rate from a very
low value to a very high value and uses the losses to estimate an optimal learning
rate. Return a [`LRFinderResult`](#).

## Keyword arguments

- `nsteps = 100`: maximum number of steps to run the learning rate finder for
- `startlr = 1e-7`: minimum learning rate
- `endlr = 10`: maximum learning rate
- `divergefactor`: stop finder early if loss goes higher than lowest loss times
    this factor
- `estimators = [Steepest(), MinDivByTen()]`: list of [`LREstimator`](#)s

"""
function lrfind(
        learner,
        dataiter = learner.data.training;
        nsteps = 100,
        startlr = 1e-7,
        endlr = 10,
        divergefactor = 4,
        estimators = [Steepest(), MinDivByTen()])
    losses = Float64[]
    lrs = Float64[]
    bestloss = Inf
    scheduler = FluxTraining.removecallback!(learner, Scheduler)  # remove current `Scheduler` so it does not interfere
    modelcheckpoint = deepcopy(cpu(learner.model))

    withfields(
        learner,
        model = modelcheckpoint,
        params = params(modelcheckpoint),
        optimizer = deepcopy(learner.optimizer)
        ) do

        FluxTraining.runepoch(learner, TrainingPhase()) do _
            for (i, batch) in zip(1:nsteps, dataiter)
                lr = startlr * (endlr / startlr) ^ (i / nsteps)
                learner.optimizer.eta = lr

                state = step!(learner, TrainingPhase(), batch)

                push!(losses, state.loss)
                push!(lrs, learner.optimizer.eta)
                bestloss = min(state.loss, bestloss)
                if state.loss > bestloss * divergefactor
                    throw(FluxTraining.CancelEpochException("Learning rate finder diverged."))
                end
            end
        end
    end

    # Restore previous `Learner` state
    isnothing(scheduler) || FluxTraining.addcallback!(learner, scheduler)
    FluxTraining.model!(learner, modelcheckpoint)
    return LRFinderResult(losses, lrs, estimators)
end

# `LREstimator`s allow giving back different suggestions for a learning rate.

"""
    abstract type LREstimator

Estimator for an optimal learning rate. Needs to implement [`estimatelr`](#).

See [`Steepest`](#) and [`MinDivBy10`](#).
"""
abstract type LREstimator end

"""
    estimatelr(::LREstimator, losses)
"""
function estimatelr end

struct Steepest <: LREstimator
    beta
end
Steepest() = Steepest(0.98)

function estimatelr(est::Steepest, losses, lrs)
    slosses = smoothvalues(losses, est.beta)
    grads = (slosses[2:end] .- slosses[1:end-1]) ./ log.(lrs[2:end] .- lrs[1:end-1])
    i = length(lrs) ÷ 3
    lr = lrs[i:end][argmax(grads[i:end])]
    return lr
end

struct MinDivByTen <: LREstimator
    beta
end
MinDivByTen() = MinDivByTen(0.98)

function estimatelr(est::MinDivByTen, losses, lrs)
    i = length(lrs) ÷ 3
    lr = lrs[i:end][argmin(smoothvalues(losses, est.beta)[i:end])] / 10
    return lr
end


LRFinderResult(losses, lrs) = LRFinderResult(losses, lrs, [Steepest(), MinDivByTen()])
LRFinderResult(losses, lrs, estimators) = LRFinderResult(
    losses,
    lrs,
    estimators,
    [estimatelr(est, losses, lrs) for est in estimators]
)

# Printing and plotting

function Base.show(io::IO, lrfindresult::LRFinderResult)
    println(io, "LRFindResult(")
    for (est, v) in zip(lrfindresult.estimators, lrfindresult.estimates)
        println(io, "    ", est, " => ", v)
    end
    print(io, ")")
end


function Makie.plot(result::LRFinderResult)
    ticks = [round((10.)^i, digits=abs(i)) for i in -10:2]
    fig = Figure()
    ax = Axis(
        title = "Learning rate finder",
        titlesize = 20,
        fig[1, 1],
        xscale = log,
        xticks = (ticks, string.(ticks)),
        xminorticks = IntervalsBetween(5),
        xminorgridvisible=true,
        ygridcolor = :white,
        xlabelsize = 14,
        ylabelsize = 14,
        ylabelcolor = :gray,
        xtickcolor= :gray,
        xticklabelcolor= :black,
        xticklabelsize= 12,
        ytickcolor= :gray,
        yticklabelcolor= :gray,
        yticklabelsize= 12,
        ylabel = "Loss",
        xlabel = "Learning rate (log)")

    lines!(
        result.lrs,
        smoothvalues(result.losses, 0.98),
        color = :black,
    )

    hidespines!(ax)

    # plot suggestions
    ls = []
    for (estim, val) in zip(result.estimators, result.estimates)
        push!(ls, vlines!(ax, [val]))

    end

    leg = Legend(
        fig[2, 1], ls,
        ["$(estim): $(round(val, sigdigits=3))" for (estim, val) in zip(result.estimators, result.estimates)],
        framevisible=false,
        labelsize=14,
        orientation = :horizontal,)
    leg.tellheight[] = true
    leg.tellwidth[] = false
    fig
end

# Utilities

function smoothvalues(xs, β)
    res = similar(xs)
    val = 0
    for (i, x) in enumerate(xs)
        val = (val * β) + (x * (1 - β))
        res[i] = val/((1)-β^i)
    end
    res
end
