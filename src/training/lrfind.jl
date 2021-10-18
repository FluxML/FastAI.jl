
"""
    LRFinderResult(lrs, losses)

Result of the learning rate finder [`lrfind`](#). Use `plot`
to visualize.
"""
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

See [`Steepest`](#) and [`MinDivByTen`](#).
"""
abstract type LREstimator end

"""
    estimatelr(::LREstimator, losses, lrs)

Estimate the optimal learning rate using `losses` and `lrs`.
"""
function estimatelr end

"""
    Steepest <: LREstimator

Estimate the optimal learning rate to be where the gradient of the loss
is the steepest, i.e. the decrease is largest.
"""
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

"""
    MinDivByTen <: LREstimator

Estimate the optimal learning rate to be value at the minimum loss divided by 10.
"""
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

function Base.show(io::IO, result::LRFinderResult)
    lrfindtextplot!(io, result)

    names = [typeof(est).name.name for est in result.estimators]
    println(io)
    println(io)
    pretty_table(
        io,
        hcat(names, result.estimates),
        header=["Estimator", "Suggestion"], tf=tf_borderless,
        alignment=[:r, :l])
end

lrfindtextplot(result::LRFinderResult) = lrfindtextplot!(stdout, result)
function lrfindtextplot!(io, result::LRFinderResult)
    p = UnicodePlots.lineplot(
        result.lrs, result.losses,
        height=10, xscale=:log10, width=displaysize(io)[2]-15)
    UnicodePlots.title!(p, "Learning rate finder result")

    for (i, estimate) in enumerate(result.estimates)
        UnicodePlots.lines!(p, estimate, maximum(result.losses), estimate, minimum(result.losses), :red)
        UnicodePlots.annotate!(p, estimate, maximum(result.losses), string(round(estimate; sigdigits=4)))
    end
    show(io, p)
    return p
end

InlineTest.@testset "LRFinderResult" begin
    res = LRFinderResult(100.:-1:1, 1:100.)
    InlineTest.@test_nowarn show(Base.DevNull(), res)
end



# Utilities

"""
    smoothvalues(xs, β)

Apply exponential smoothing with parameter `β` to vector `xs`.
"""
function smoothvalues(xs, β)
    res = similar(xs)
    val = 0
    for (i, x) in enumerate(xs)
        val = (val * β) + (x * (1 - β))
        res[i] = val/((1)-β^i)
    end
    res
end
