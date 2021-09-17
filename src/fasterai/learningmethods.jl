# ## Computer vision

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

# ---

function ImageSegmentation(
		blocks::Tuple{<:Image{N},<:Mask{N}},
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


# ---


function ImageKeypointRegression(
        blocks::Tuple{<:Image{N},<:Keypoints{N}},
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


# ## Tabular

function TabularClassificationSingle(
        blocks::Tuple{<:TableRow, <:Label},
        data)
    tabledata, targetdata = data
    tabledata isa TableDataset || error("`data` needs to be a tuple of a `TableDataset` and targets")

    return BlockMethod(
        blocks,
        (
            setup(TabularPreprocessing, blocks[1], tabledata),
            OneHot()
        )
    )
end

"""
    TabularClassificationSingle(blocks, data)

Learning method for single-label tabular classification. Continuous columns are
normalized and missing values are filled, categorical columns are label encoded
taking into account any missing values which might be present. The target value
is predicted from `classes`. `blocks` should be an input and target block
`(TableRow(...), Label(...))`.

    TabularClassificationSingle(classes, tabledata [; catcols, contcols])

Construct learning method with `classes` to classify into and a `TableDataset`
`tabledata`. The column names can be passed in or guessed from the data.
"""
function TabularClassificationSingle(
        classes::AbstractVector,
        tabledata::TableDataset;
        catcols = nothing,
        contcols = nothing)

    blocks = (
        setup(TableRow, tabledata; catcols = catcols, contcols = contcols),
        Label(classes)
    )
    return TabularClassificationSingle(blocks, (tabledata, nothing))
end

# ---

function TabularRegression(
        blocks::Tuple{<:TableRow, <:Continuous},
        data)
    tabledata, targetdata = data
    tabledata isa TableDataset || error("`data` needs to be a tuple of a `TableDataset` and targets")
    return BlockMethod(
        blocks,
        (setup(TabularPreprocessing, blocks[1], tabledata),),
        outputblock=blocks[2]
    )
end

"""
    TabularRegression(blocks, data)

Learning method for tabular regression. Continuous columns are
normalized and missing values are filled, categorical columns are label encoded
taking into account any missing values which might be present.
 `blocks` should be an input and target block `(TableRow(...), Continuous(...))`.

    TabularRegression(n, tabledata [; catcols, contcols])

Construct learning method with `classes` to classify into and a `TableDataset`
`tabledata`. The column names can be passed in or guessed from the data. The
regression target is a vector of `n` values.
"""
function TabularRegression(
        n::Int,
        tabledata::TableDataset;
        catcols = nothing,
        contcols = nothing)
    blocks = (
        setup(TableRow, tabledata; catcols=catcols, contcols=contcols),
        Continuous(n)
    )
    return TabularRegression(blocks, (tabledata, nothing))
end
