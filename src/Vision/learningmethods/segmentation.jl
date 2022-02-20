
function ImageSegmentation(
		blocks::Tuple{<:Image{N},<:Mask{N}},
        data=nothing;
		size=ntuple(i -> 128, N),
		aug_projections=DataAugmentation.Identity(),
		aug_image=DataAugmentation.Identity(),
        C=RGB{N0f8},
        computestats=false,
	) where N
	return SupervisedMethod(
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

Learning method for image segmentation. Images are
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

registerlearningmethod!(FASTAI_METHOD_REGISTRY, ImageSegmentation, (Image, Mask))


# ## Tests

@testset "ImageSegmentation [method]" begin
    @testset "2D" begin
        method = ImageSegmentation((16, 16), 1:4)
        testencoding(getencodings(method), getblocks(method).sample)
        DLPipelines.checkmethod_core(method)
        @test_nowarn methodlossfn(method)
        @test_nowarn methodmodel(method, Models.xresnet18())
        @testset "Show backends" begin
            @testset "ShowText" begin
                FastAI.test_method_show(method, ShowText(Base.DevNull()))
            end
        end
    end
    @testset "3D" begin
        method = SupervisedMethod(
            (Image{3}(), Mask{3}(1:4)),
            (
                ProjectiveTransforms((16, 16, 16), inferencefactor=8),
                ImagePreprocessing(),
                FastAI.OneHot()
            )
        )
        testencoding(getencodings(method), getblocks(method).sample)
        DLPipelines.checkmethod_core(method)
        @test_nowarn methodlossfn(method)
    end

end
