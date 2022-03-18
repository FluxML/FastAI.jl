
function ImageKeypointRegression(
        blocks::Tuple{<:Image{N},<:Keypoints{N}},
        data=nothing;
        size=ntuple(i -> 128, N),
		aug_projections=DataAugmentation.Identity(),
		aug_image=DataAugmentation.Identity(),
        C=RGB{N0f8},
        computestats=false,
    ) where N
    return SupervisedTask(
        blocks,
        (
			ProjectiveTransforms(size; augmentations=aug_projections),
			getimagepreprocessing(data, computestats; C=C, augmentations=aug_image),
            KeypointPreprocessing(size),
        )
    )
end


"""
    ImageKeypointRegression(size, nkeypoints; kwargs...)

Learning task for regressing a set of `nkeypoints` keypoints from
images. Images are resized to `size` and a class is predicted for every pixel.
"""
function ImageKeypointRegression(size::NTuple{N,Int}, nkeypoints::Int; kwargs...) where N
    blocks = (Image{N}(), Keypoints{N}((nkeypoints,)))
    return ImageKeypointRegression(blocks; size = size, kwargs...)
end

registerlearningtask!(FASTAI_METHOD_REGISTRY, ImageKeypointRegression, (Image, Keypoints))


# ## Tests

@testset "ImageKeypointRegression [task]" begin
    task = ImageKeypointRegression((16, 16), 10)
    FastAI.checktask_core(task)
    @testset "Show backends" begin
        @testset "ShowText" begin
            FastAI.test_task_show(task, ShowText(Base.DevNull()))
        end
    end
end
