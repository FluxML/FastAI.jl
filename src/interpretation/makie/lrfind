
function Makie.plot(result::FastAI.LRFinderResult)
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

    return fig
end
