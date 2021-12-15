
function ImageClassificationSingle(
		blocks::Tuple{<:Image{N},<:Label},
        data=nothing;
		size=ntuple(i -> 128, N),
		aug_projections=DataAugmentation.Identity(),
		aug_image=DataAugmentation.Identity(),
        C=RGB{N0f8},
        computestats=false,
	) where N
	return BlockMethod(
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

Learning method for single-label image classification. Images are
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

registerlearningmethod!(FASTAI_METHOD_REGISTRY, ImageClassificationSingle, (Image, Label))

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
	return BlockMethod(
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

Learning method for multi-label image classification. Images are
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


registerlearningmethod!(FASTAI_METHOD_REGISTRY, ImageClassificationMulti, (Image, LabelMulti))


# ## Tests

@testset "ImageClassificationSingle [method]" begin
    method = ImageClassificationSingle((16, 16), [1, 2])
    testencoding(method.encodings, method.blocks)
    DLPipelines.checkmethod_core(method)
    @test_nowarn methodlossfn(method)
    @test_nowarn methodmodel(method, Models.xresnet18())

    @testset "`encodeinput`" begin
        image = rand(RGB, 32, 48)

        xtrain = encodeinput(method, Training(), image)
        @test size(xtrain) == (16, 16, 3)
        @test eltype(xtrain) == Float32

        xinference = encodeinput(method, Inference(), image)
        @test size(xinference) == (16, 24, 3)
        @test eltype(xinference) == Float32
    end
    @testset "`encodetarget`" begin
        category = 1
        y = encodetarget(method, Training(), category)
        @test y ≈ [1, 0]
        # depends on buffered interface for `BlockMethod`s and `Encoding`s
        #encodetarget!(y, method, Training(), 2)
        #@test y ≈ [0, 1]
    end
    @testset "Show backends" begin
        @testset "ShowText" begin
            #@test_broken FastAI.test_method_show(method, ShowText(Base.DevNull()))
        end
    end

    @testset "blockmodel" begin
        method = ImageClassificationSingle((Image{2}(), Label(1:2)))
        @test_nowarn methodmodel(method)
    end
end

@testset "ImageClassificationMulti [method]" begin

    method = ImageClassificationMulti((16, 16), [1, 2])

    testencoding(method.encodings, method.blocks)
    DLPipelines.checkmethod_core(method)
    @test_nowarn methodlossfn(method)
    @test_nowarn methodmodel(method, Models.xresnet18())
    @testset "Show backends" begin
        @testset "ShowText" begin
            FastAI.test_method_show(method, ShowText(Base.DevNull()))
        end
    end
end
