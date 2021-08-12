


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

Contruct a model for tabular classification or regression. `backbone` should 
either be `nothing` or a tuple of categoricalbackbone, continuousbackbone, 
and a classifierbackbone, with the first two taking in batches of corresponding 
row value matrices.
"""

# function blockmodel(
#         inblock::EncodedTableRow{M, N}, 
#         outblock::Union{Continuous, OneHotTensor{0}}, 
#         backbone=nothing;) where {M, N}
#     outsz = outblock isa Continuous ? outblock.size : length(outblock.classes)
#     if isnothing(backbone)
#         TabularModel(inblock.catcols, N, outsz; catdict = inblock.categorydict)
#     else
#         TabularModel(backbone[1], backbone[2], backbone[3])
#     end
# end



