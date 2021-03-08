

struct LRFinderResult{T}
    losses::Vector{T}
    lrs::Vector{T}
    steepest::T
    mindiv10::T
end

function LRFinderResult(losses::Vector{T}, lrs) where T
    return LRFinderResult(losses, lrs, zero(T), zero(T))
end


mutable struct LRFinderPhase <: FluxTraining.Phases.AbstractTrainingPhase
    startlr
    endlr
    steps
    β
    divergefactor
    result
end

function LRFinderPhase(; startlr = 1e-7, endlr = 10, steps = 100, β = 0.02, divergefactor = 4, result = nothing)
    return LRFinderPhase(startlr, endlr, steps, β, divergefactor, result)
end


function FluxTraining.fitepochphase!(
        learner::Learner,
        phase::LRFinderPhase)

    dataiter = FluxTraining.getdataiter(TrainingPhase(), learner)
    if dataiter === nothing
        throw(CancelEpochException("No data found for phase $(typeof(phase))"))
    end

    metric = SmoothLoss(phase.β)
    metricscb = Metrics(metric)
    losses = Float32[]
    lrs = Float32[]

    withfields(
        learner,
        model = (FluxTraining.model!, deepcopy(cpu(learner.model))),
        optimizer = deepcopy(learner.optimizer)
        ) do

        withcallbacks(learner, metricscb, Scheduler(Dict())) do
            #schedule = Schedule([0, phase.steps], [phase.startlr, phase.endlr], Animations.expin(100000000))
            #setschedules!(learner, phase, LearningRate => schedule)
            bestloss = Inf

            handle(EpochBegin(), learner, phase)
            for (i, batch) in zip(1:phase.steps, dataiter)
                lr = phase.startlr * (phase.endlr / phase.startlr) ^ (i / phase.steps)
                learner.optimizer.eta = lr
                FluxTraining.fitbatch!(learner, batch, phase)

                loss = FluxTraining.stepvalue(metric)
                bestloss = min(bestloss, loss)
                push!(losses, loss)
                push!(lrs, lr)

                # Stop if loss is diverging
                if loss > bestloss * phase.divergefactor
                    break
                end
            end
            handle(EpochEnd(), learner, phase)
        end
    end

    phase.result = LRFinderResult(losses, lrs)
end



function plotlrfind(lrs, losses)
    f = Figure(resolution = (700, 400))
    f[1, 1] = ax = Axis(f)

    xtickvalues = (10.) .^(-10:10)
    ax.xlabel = "Learning rate"
    ax.xticks = log.(xtickvalues)
    ax.xtickformat[] = vals -> string.(xtickvalues)
    ax.xticklabelsize = 10.

    ax.ylabel = "Loss"
    ax.yticks = []
    ax.ytickformat[] = vals -> string.(log.(vals))
    ax.yticklabelsize = 10.


    AbstractPlotting.lines!(ax, log.(lrs), log.(losses), axis = (xticks = LinearTicks(10),))
    return f
end

plotlrfind(phase::LRFinderPhase) = plotlrfind(phase.result)
plotlrfind(lrresult::LRFinderResult) = plotlrfind(lrresult.lrs, lrresult.losses)
