

function getimagepreprocessing(data, computestats::Bool; kwargs...)
    if isnothing(data) && computestats
        error("If `computestats` is `true`, you have to pass in a data container `data`.")
    end
    return if computestats
        setup(ImagePreprocessing, Image{2}(), data; kwargs...)
    else
        ImagePreprocessing(; kwargs...)
    end
end
