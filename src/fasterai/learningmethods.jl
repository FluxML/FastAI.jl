# ## Computer vision

function ImageClassificationSingle(
		blocks::Tuple{<:Image{N},<:Label};
		size=ntuple(i -> 128, N),
		aug_projections=DataAugmentation.Identity(),
		aug_image=DataAugmentation.Identity(),
	) where N
	return BlockMethod(
		blocks,
		(
			ProjectiveTransforms(size; augmentations=aug_projections),
			ImagePreprocessing(),
        	OneHot()
		)
	)
end

"""
    ImageClassificationSingle(size, classes; kwargs...)

Learning method for single-label image classification. Images are
resized to `size` and classified into one of `classes`.

Use [`ImageClassificationMulti`](#) for the multi-class setting.
"""
function ImageClassificationSingle(size::NTuple{N,Int}, classes::AbstractVector; kwargs...) where N
    blocks = (Image{N}(), Label(classes))
    return ImageClassificationSingle(blocks, size=size)
end

registerlearningmethod!(FASTAI_METHOD_REGISTRY, ImageClassificationSingle, (Image, Label))

# ---

function ImageClassificationMulti(
		blocks::Tuple{<:Image{N},<:LabelMulti};
		size=ntuple(i -> 128, N),
		aug_projections=DataAugmentation.Identity(),
		aug_image=DataAugmentation.Identity(),
	) where N
	return BlockMethod(
		blocks,
		(
			ProjectiveTransforms(size; augmentations=aug_projections),
			ImagePreprocessing(augmentations=aug_image),
        	OneHot()
		)
	)
end


"""
    ImageClassificationMulti(size, classes; kwargs...)

Learning method for single-label image classification. Images are
resized to `size` and classified into multiple of `classes`.

Use [`ImageClassificationMulti`](#) for the single-class setting.
"""
function ImageClassificationMulti(size::NTuple{N,Int}, classes::AbstractVector; kwargs...) where N
    blocks = (Image{N}(), LabelMulti(classes))
    return ImageClassificationMulti(blocks; size=size, kwargs...)
end


registerlearningmethod!(FASTAI_METHOD_REGISTRY, ImageClassificationMulti, (Image, LabelMulti))

# ---

function ImageSegmentation(
		blocks::Tuple{<:Image{N},<:Mask{N}};
		size=ntuple(i -> 128, N),
		aug_projections=DataAugmentation.Identity(),
		aug_image=DataAugmentation.Identity(),
	) where N
	return BlockMethod(
		blocks,
		(
			ProjectiveTransforms(size; augmentations=aug_projections),
			ImagePreprocessing(augmentations=aug_image),
        	OneHot()
		)
	)
end


"""
    ImageSegmentation(size, classes; kwargs...)

Learning method for image segmentation. Images are
resized to `size` and a class is predicted for every pixel.
"""
function ImageSegmentation(size::NTuple{N,Int}, classes::AbstractVector; kwargs...) where N
    blocks = (Image{N}(), Mask{N}(classes))
    return ImageSegmentation(blocks; size, kwargs...)
end

registerlearningmethod!(FASTAI_METHOD_REGISTRY, ImageSegmentation, (Image, Mask))


# ---


function ImageKeypointRegression(
        blocks::Tuple{<:Image{N},<:Keypoints{N}};
        size=ntuple(i -> 128, N),
		aug_projections=DataAugmentation.Identity(),
		aug_image=DataAugmentation.Identity(),
    ) where N
    return BlockMethod(
        blocks,
        (
			ProjectiveTransforms(size; augmentations=aug_projections),
			ImagePreprocessing(augmentations=aug_image),
            KeypointPreprocessing(size),
        )
    )
end


"""
    ImageKeypointRegression(size, nkeypoints; kwargs...)

Learning method for regressing a set of `nkeypoints` keypoints from
images. Images are resized to `size` and a class is predicted for every pixel.
"""
function ImageKeypointRegression(size::NTuple{N,Int}, nkeypoints::Int; kwargs...) where N
    blocks = (Image{N}(), Keypoints{N}((nkeypoints,)))
    return ImageKeypointRegression(blocks; size = size, kwargs...)
end

registerlearningmethod!(FASTAI_METHOD_REGISTRY, ImageKeypointRegression, (Image, Keypoints))
