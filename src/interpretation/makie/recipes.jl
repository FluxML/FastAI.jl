
"""
    imageaxis(f)

Create a `Makie.Axis` with no interactivity, decorations and aspect distortion
suitable for showing images on.
"""
function imageaxis(f; kwargs...)
    ax = Makie.Axis(f; kwargs...)
    ax.aspect = DataAspect()
    ax.xzoomlock = true
    ax.yzoomlock = true
    ax.xrectzoom = false
    ax.yrectzoom = false
    ax.panbutton = nothing
    ax.xpanlock = true
    ax.ypanlock = true
    ax.bottomspinevisible = false
    ax.leftspinevisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    hidedecorations!(ax)

    return ax
end


@recipe(PlotImage, image) do scene
    Attributes(
        alpha = 1,
        interpolate = false,
    )
end

function Makie.plot!(plot::PlotImage)
    im = plot[:image]
    rim = @lift alphacolor.(copy(rotr90($im)), $(plot.attributes[:alpha]))
    image!(plot, rim; plot.attributes...)
    return plot
end


@recipe(PlotMask, mask, classes) do scene
    Attributes()
end

function Makie.plot!(plot::PlotMask; kwargs...)
    mask = plot[:mask]
    classes = try
        classes = plot[:classes]
    catch
        classes = @lift unique($mask)
    end
    im = @lift maskimage($mask, $classes)
    plotimage!(plot, im; alpha = 1, plot.attributes...)
    return plot
end
