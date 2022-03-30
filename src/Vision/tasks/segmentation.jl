
function ImageSegmentation(
		blocks::Tuple{<:Image{N},<:Mask{N}},
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
        	OneHot()
		)
	)
end


"""
    ImageSegmentation(size, classes; kwargs...)

Learning task for image segmentation. Images are
resized to `size` and a class is predicted for every pixel.

## Keyword arguments

- `computestats = false`: Whether to compute image statistics on dataset `data` or use
    default ImageNet stats.
- `aug_projections = `[`DataAugmentation.Identity`](#): augmentation to apply during
  [`ProjectiveTransforms`](#) (resizing and cropping)
- `aug_image = `[`DataAugmentation.Identity`](#): pixel-level augmentation to apply during
  [`ImagePreprocessing`](#)
- `C = RGB{N0f8}`: Color type images are converted to before further processing. Use `Gray{N0f8}`
    for grayscale images.
"""
function ImageSegmentation(size::NTuple{N,Int}, classes::AbstractVector; kwargs...) where N
    blocks = (Image{N}(), Mask{N}(classes))
    return ImageSegmentation(blocks; size=size, kwargs...)
end

registerlearningtask!(FASTAI_METHOD_REGISTRY, ImageSegmentation, (Image, Mask))

_tasks["imagesegmentation"] = (
    id = "vision/imagesegmentation",
    name = "Image segmentation",
    constructor = ImageKeypointRegression,
    blocks = (Image, Mask),
    category = "supervised",
    description = """
        Semantic segmentation task in which every pixel in an image is
        classified.
        """,
    package=@__MODULE__,
)




# ## Tests

@testset "ImageSegmentation [task]" begin
    @testset "2D" begin
        task = ImageSegmentation((16, 16), 1:4)
        testencoding(getencodings(task), getblocks(task).sample)
        FastAI.checktask_core(task)
        @test_nowarn tasklossfn(task)
        @test_nowarn taskmodel(task, Models.xresnet18())
        @testset "Show backends" begin
            @testset "ShowText" begin
                FastAI.test_task_show(task, ShowText(Base.DevNull()))
            end
        end
    end
    @testset "3D" begin
        task = SupervisedTask(
            (Image{3}(), Mask{3}(1:4)),
            (
                ProjectiveTransforms((16, 16, 16), inferencefactor=8),
                ImagePreprocessing(),
                FastAI.OneHot()
            )
        )
        testencoding(getencodings(task), getblocks(task).sample)
        FastAI.checktask_core(task)
        @test_nowarn tasklossfn(task)
    end

end
