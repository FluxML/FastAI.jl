



function showblock!(ax, ::ShowMakie, block::Image{2}, obs)
    plotimage!(ax, obs)
end
function showblock!(ax, ::ShowMakie, block::Mask{2}, obs)
    plotmask!(ax, obs, block.classes)
end


function showblock!(ax, ::ShowMakie, block::Keypoints{2}, obs)
    h = maximum(first.(obs))
    ks = [SVector(x, h-y) for (y, x) in obs]
    MakieCore.scatter!(ax, ks)
end

function showblock!(ax, ::ShowMakie, block::Bounded{2, <:Keypoints{2}}, obs)
    h, w = block.size
    ks = [SVector(x, h-y) for (y, x) in obs]
    MakieCore.xlims!(ax, 0, w)
    MakieCore.ylims!(ax, 0, h)
    MakieCore.scatter!(ax, ks)
end

# ## Helpers


@recipe(PlotImage, image) do scene
    MakieCore.Attributes(
        alpha = 1,
        interpolate = false,
    )
end

function MakieCore.plot!(plot::PlotImage)
    im = plot[:image]
    rim = @map alphacolor.(copy(rotr90(&im)), &(plot.attributes[:alpha]))
    MakieCore.image!(plot, rim; plot.attributes...)
    return plot
end


@recipe(PlotMask, mask, classes) do scene
    MakieCore.Attributes()
end

function MakieCore.plot!(plot::PlotMask; kwargs...)
    mask = plot[:mask]
    classes = try
        classes = plot[:classes]
    catch
        classes = @map unique(&mask)
    end
    im = @map _maskimage(&mask, &classes)
    plotimage!(plot, im; alpha = 1, plot.attributes...)
    return plot
end
