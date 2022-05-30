
function ImageClassificationSingle(
		blocks::Tuple{<:Image{N},<:Label},
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
    ImageClassificationSingle(size, classes; kwargs...)
    ImageClassificationSingle(blocks[, data]; kwargs...)

Learning task for single-label image classification. Images are
resized to `size` and classified into one of `classes`.

Use [`ImageClassificationMulti`](#) for the multi-class setting.

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
function ImageClassificationSingle(size::NTuple{N,Int}, classes::AbstractVector; kwargs...) where N
    blocks = (Image{N}(), Label(classes))
    return ImageClassificationSingle(blocks, size=size)
end


_tasks["imageclfsingle"] = (
    id = "vision/imageclfsingle",
    name = "Image classification (single-label)",
    constructor = ImageClassificationSingle,
    blocks = (Image, Label),
    category = "supervised",
    description = """
        Single-label image classification task where every image has a single
        class label associated with it.
        """,
    package=@__MODULE__,
)

# ---

function ImageClassificationMulti(
		blocks::Tuple{<:Image{N},<:LabelMulti},
        data = nothing;
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
    ImageClassificationMulti(size, classes; kwargs...)

Learning task for multi-label image classification. Images are
resized to `size` and classified into multiple of `classes`.

Use [`ImageClassificationSingle`](#) for the single-class setting.

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
function ImageClassificationMulti(size::NTuple{N,Int}, classes::AbstractVector; kwargs...) where N
    blocks = (Image{N}(), LabelMulti(classes))
    return ImageClassificationMulti(blocks; size=size, kwargs...)
end



_tasks["imageclfmulti"] = (
    id = "vision/imageclfmulti",
    name = "Image classification (multi-label)",
    constructor = ImageClassificationMulti,
    blocks = (Image, LabelMulti),
    category = "supervised",
    description = """
        Multi-label image classification task where every image can
        have multiple class labels associated with it.
        """,
    package=@__MODULE__,
)



# ## Tests

@testset "ImageClassificationSingle [task]" begin
    task = ImageClassificationSingle((16, 16), [1, 2])
    testencoding(getencodings(task), getblocks(task).sample)
    FastAI.checktask_core(task)
    @test_nowarn tasklossfn(task)
    @test_nowarn taskmodel(task, Models.xresnet18())

    @testset "`encodeinput`" begin
        image = rand(RGB, 32, 48)

        xtrain = encodeinput(task, Training(), image)
        @test size(xtrain) == (16, 16, 3)
        @test eltype(xtrain) == Float32

        xinference = encodeinput(task, Inference(), image)
        @test size(xinference) == (16, 24, 3)
        @test eltype(xinference) == Float32
    end
    @testset "`encodetarget`" begin
        category = 1
        y = encodetarget(task, Training(), category)
        @test y ≈ [1, 0]
        # depends on buffered interface for `BlockTask`s and `Encoding`s
        #encodetarget!(y, task, Training(), 2)
        #@test y ≈ [0, 1]
    end
    @testset "Show backends" begin
        @testset "ShowText" begin
            #@test_broken FastAI.test_task_show(task, ShowText(Base.DevNull()))
        end
    end

    @testset "blockmodel" begin
        task = ImageClassificationSingle((Image{2}(), Label(1:2)))
        @test_nowarn taskmodel(task)
    end
end

@testset "ImageClassificationMulti [task]" begin

    task = ImageClassificationMulti((16, 16), [1, 2])

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
