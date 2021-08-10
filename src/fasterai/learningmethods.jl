# ## Computer vision

function ImageClassificationSingle(
		blocks::Tuple{<:Image{N}, <:Label};
		sz = ntuple(i -> 128, N),
		aug_projections = DataAugmentation.Identity(),
		aug_image = DataAugmentation.Identity(),
	) where N
	return BlockMethod(
		blocks,
		(
			ProjectiveTransforms(sz; augmentations=aug_projections),
			ImagePreprocessing(),
        	OneHot()
		)
	)
end

registerlearningmethod!(FASTAI_METHOD_REGISTRY, ImageClassificationSingle, (Image, Label))


function ImageClassificationMulti(
		blocks::Tuple{<:Image{N}, <:LabelMulti};
		sz = ntuple(i -> 128, N),
		aug_projections = DataAugmentation.Identity(),
		aug_image = DataAugmentation.Identity(),
	) where N
	return BlockMethod(
		blocks,
		(
			ProjectiveTransforms(sz; augmentations=aug_projections),
			ImagePreprocessing(augmentations=aug_image),
        	OneHot()
		)
	)
end

registerlearningmethod!(FASTAI_METHOD_REGISTRY, ImageClassificationMulti, (Image, LabelMulti))


function ImageSegmentation(
		blocks::Tuple{<:Image{N}, <:Mask{N}};
		sz = ntuple(i -> 128, N),
		aug_projections = DataAugmentation.Identity(),
		aug_image = DataAugmentation.Identity(),
	) where N
	return BlockMethod(
		blocks,
		(
			ProjectiveTransforms(sz; augmentations=aug_projections),
			ImagePreprocessing(augmentations=aug_image),
        	OneHot()
		)
	)
end

registerlearningmethod!(FASTAI_METHOD_REGISTRY, ImageSegmentation, Tuple{Image, Mask})
