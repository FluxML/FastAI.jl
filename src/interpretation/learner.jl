# High-level plotting functions for use with `BlockMethod` and a `Learner`


"""
    showoutputs(method, learner[; n = 4, validation = true])

Run a trained model in `learner` on `n` samples and visualize the
outputs.
"""
function showoutputs(method::BlockMethod, learner::Learner; n=4, validation=true, backend = default_showbackend())
    cb = FluxTraining.getcallback(learner, ToDevice)
    devicefn = isnothing(cb) ? identity : cb.movedatafn
    backfn = isnothing(cb) ? identity : cpu

    xs, ys = getbatch(learner; n = n, validation = validation)
    ŷs = learner.model(devicefn(xs)) |> backfn
    return showoutputbatch(backend, method, (xs, ys), ŷs)
end
