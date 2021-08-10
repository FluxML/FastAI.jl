


"""
    blockmodel(inblock, outblock, backbone)

From a `backbone` model, construct a model suitable for learning
a mapping from `inblock` to `outblock`.
"""
function blockmodel end



"""
    blockmodel(inblock::ImageTensor{N}, outblock::OneHotTensor{0}, backbone)
    blockmodel(inblock::ImageTensor{N}, outblock::OneHotTensorMulti{0}, backbone)

Construct a model for N-dimensional image classification. `backbone` should
be a convolutional feature extractor taking in batches of image tensors with
`inblock.nch` color channels.
"""
function blockmodel(
        inblock::ImageTensor{N},
        outblock::Union{OneHotTensor{0}, OneHotTensorMulti{0}},
        backbone) where N
    outsz = Flux.outputsize(backbone, (ntuple(_ -> 256, N)..., inblock.nchannels, 1))
    outch = outsz[end-1]
    head = Models.visionhead(outch, length(outblock.classes), p = 0.)
    return Chain(backbone, head)
end


"""
    blockmodel(inblock::ImageTensor{N}, outblock::OneHotTensor{N}, backbone; kwargs...)

Construct a model for N-dimensional image segmentation. `backbone` should
be a convolutional feature extractor taking in batches of image tensors with
`inblock.nch` color channels. Keyword arguments are passed to [`UNetDynamic`](#).
"""
function blockmodel(inblock::ImageTensor{N}, outblock::OneHotTensor{N}, backbone; kwargs...) where N
    return UNetDynamic(
        backbone,
        (ntuple(_ -> 256, N)..., inblock.nchannels, 1),
        length(outblock.classes);
        kwargs...)
end


"""
    blockmodel(inblock::ImageTensor{N}, outblock::Keypoints{N}, backbone)

Construct a model for image to keypoint regression. `backbone` should
be a convolutional feature extractor taking in batches of image tensors with
`inblock.nch` color channels.
"""
function blockmodel(inblock::ImageTensor{N}, outblock::KeypointTensor{N}, backbone) where N
    outsz = Flux.outputsize(backbone, (ntuple(_ -> 256, N)..., inblock.nchannels, 1))
    outch = outsz[end-1]
    head = Models.visionhead(outch, prod(outblock.sz)*N, p = 0.)
    return Chain(backbone, head)
end


function blockmodel(
        inblock::RawCategoricalBlock, 
        ::Union{RawContinuousBlock, OneHotCols},
        embszs=nothing)
    embszs = isnothing(embszs) ? Models.get_emb_sz(inblock.categorydict, inblock.columns; sz_dict=nothing) : embszs
    Models.embeddingbackbone(embszs)
end

function blockmodel(
        ::RawContinuousBlock{N, T, M}, 
        ::Union{RawContinuousBlock, OneHotCols},
        backbone=nothing) where {N, T, M}
    Models.continuousbackbone(N)
end

function blockmodel(
        inblock::Tuple{RawCategoricalBlock{N, T, M}, RawContinuousBlock{O, T, M}},
        outblock::Union{RawContinuousBlock{P, T, M}, OneHotCols{P, T, M}},
        backbone;
        embszs = nothing) where {N, O, P, T, M}
    catbackbone = blockmodel(inblock[1], outblock, embszs)
    contbackbone = blockmodel(inblock[2], outblock, nothing)
    Models.TabularModel(
        catbackbone, 
        contbackbone, 
        n_cat=N,
        n_cont=O,
        out_sz=outblock isa OneHotCols ? P+1 : P)
end



