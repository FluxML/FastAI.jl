

const _models = Dict{String, Any}()

function cnn_variants(; nfeatures = :, hasweights = false)
    variants = Pair{String, ModelVariant}[]

    hasweights && push!(variants, "imagenet_1k" => ModelVariant(
            input=ImageTensor{2}(3),
            # TODO: use actual ImageNet classes
            output=FastAI.OneHotLabel{Int}(1:1000),
        ))
    push!(variants, "classifier" => ModelVariant(
        make_cnn_classifier,
        ImageTensor{2},
        FastAI.OneHotTensor{0},
    ))
    push!(variants, "backbone" => ModelVariant(
        make_cnn_backbone,
        ImageTensor{2},
        ConvFeatures{2}(nfeatures),
    ))

    return variants
end

function make_cnn_classifier(model, input::ImageTensor, output::OneHotTensor{0})
    backbone = _backbone_with_channels(model.layers[1], input.nchannels)
    head = _head_with_classes(model.layers[2], length(output.classes))
    return Chain(backbone, head)
end

function make_cnn_backbone(model, input::ImageTensor{N}, output::ConvFeatures{N}) where N
    backbone = _backbone_with_channels(model.layers[1], input.nchannels)
    return backbone
end

function _backbone_with_channels(backbone, n)
    layer = backbone.layers[1].layers[1]
    layer isa Conv || throw(ArgumentError(
        """To change the number of input channels,
        `backbone.layers[1].layers[1]` must be a `Conv` layer."""))

    sz = size(layer.weight)
    ks = sz[begin:end-2]
    in_, out = sz[end-1:end]
    in_ == n && return backbone

    layer = @set layer.weight = Flux.kaiming_normal(Random.GLOBAL_RNG, ks..., n, out)
    return @set backbone.layers[1].layers[1] = layer
end

function _head_with_classes(head, n)
    head.layers[end] isa Dense || throw(ArgumentError(
        """To change the number of output classes,
        the last layer in head must be a `Dense` layer."""))
    c, f = size(head[end].weight)
    if c == n
        # Already has correct number of classes
        head
    else
        @set head.layers[end] = Dense(f, n)
    end
end

function metalhead_loadfn(modelfn, args...)
    return function (checkpoint; kwargs...)
        return modelfn(args...; pretrain=!isnothing(checkpoint), kwargs...)
    end
end

for depth in (18,)
    hasweights = true
    nfeatures = 512
    id = "metalhead/resnet$depth"
    _models[id] = (;
        id = id,
        variants = cnn_variants(; hasweights, nfeatures),
        checkpoints = hasweights ? ["imagenet1k"] : String[],
        backend = :flux,
        loadfn = metalhead_loadfn(Metalhead.ResNet, depth)
    )
end


@testset "Model variants" begin
    @testset "make_cnn_classifier" begin
        m = Metalhead.ResNet(18)
        clf = make_cnn_classifier(m, ImageTensor{2}(3), FastAI.OneHotLabel{Int}(1:10))
        @test Flux.outputsize(clf, (256, 256, 3, 1)) == (10, 1)

        clf2 = make_cnn_classifier(m, ImageTensor{2}(5), FastAI.OneHotLabel{Int}(1:100))
        @test Flux.outputsize(clf2, (256, 256, 5, 1)) == (100, 1)
    end

    @testset "make_cnn_backbone" begin
        m = Metalhead.ResNet(18)
        clf = make_cnn_backbone(m, ImageTensor{2}(10), ConvFeatures{2}(512))
        @test Flux.outputsize(clf, (256, 256, 10, 1)) == (8, 8, 512, 1)

    end
end
