
function Makie.plot(result::FastAI.LRFinderResult; theme = Makie.theme_light())
    fig = Makie.with_theme(theme) do
        ticks = [round((10.)^i, digits=abs(i)) for i in -10:2]
        fig = Makie.Figure()
        ax = Makie.Axis(
            title = "Learning rate finder",
            titlesize = 20,
            fig[1, 1],
            xscale = log,
            xticks = (ticks, string.(ticks)),
            xminorticks = Makie.IntervalsBetween(5),
            xminorgridvisible=true,
            xlabelsize = 14,
            ylabelsize = 14,
            xticklabelsize= 12,
            yticklabelsize= 12,
            ylabel = "Loss",
            xlabel = "Learning rate (log)")

        Makie.lines!(
            result.lrs,
            linewidth=3.,
            FastAI.smoothvalues(result.losses, 0.98),
            color = :green,
        )

        Makie.hidespines!(ax)

        # plot suggestions
        ls = []
        for (estim, val) in zip(result.estimators, result.estimates)
            push!(ls, Makie.vlines!(ax, [val]))
        end

        leg = Makie.Legend(
            fig[2, 1], ls,
            ["$(nameof(typeof(estim))): $(round(val, sigdigits=3))" for (estim, val) in zip(result.estimators, result.estimates)],
            framevisible=false,
            labelsize=14,
            orientation = :horizontal,)
        leg.tellheight[] = true
        leg.tellwidth[] = false
        fig
    end
end
