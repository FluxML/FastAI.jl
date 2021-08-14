


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

"""
    blockmodel(inblock::TableRow{M, N}, outblock::Union{Continuous, OneHotTensor{0}}, backbone=nothing) where {M, N}

Contruct a model for tabular classification or regression. `backbone` should be a 
Dict of categoricalbackbone, continuousbackbone, and a finalclassifier layer, with 
the first two taking in batches of corresponding row value matrices.
"""

function blockmodel(
        inblock::EncodedTableRow{M, N}, 
        outblock::Union{Continuous, OneHotTensor{0}}, 
        backbone=Dict()) where {M, N}
    
    backbone = Dict{Any, Any}((backbonetype => layer) for (backbonetype, layer) in collect(backbone))
    backbonekeys = keys(backbone)
    if !(:categoricalbackbone in backbonekeys)
        embedszs = FastAI.Models.get_emb_sz(
            Dict(col => length(classes) for (col, classes) in collect(inblock.categorydict)),
            inblock.catcols)
        backbone[:categoricalbackbone] = Models.tabular_embedding_backbone(embedszs)
    end
    if !(:continuousbackbone in backbonekeys)
        backbone[:continuousbackbone] = Models.tabular_continuous_backbone(N)
    end
    if (:finalclassifier in backbonekeys)
        TabularModel(backbone[:categoricalbackbone], backbone[:continuousbackbone], backbone[:finalclassifier])
    else
        outsz = outblock isa Continuous ? outblock.size : length(outblock.classes)
        TabularModel(backbone[:categoricalbackbone], backbone[:continuousbackbone]; outsz=outsz)
    end
end



