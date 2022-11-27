
const _models = Dict{String, Any}()

function cnn_variants(; nfeatures = :, hasweights = false)
    variants = Pair{String, ModelVariant}[]

    hasweights && push!(variants,
          "imagenet_1k" => ModelVariant(xblock = ImageTensor{2}(3),
                                        # TODO: use actual ImageNet classes
                                        yblock = FastAI.OneHotLabel{Int}(1:1000)))
    push!(variants,
          "classifier" => ModelVariant(make_cnn_classifier,
                                       ImageTensor{2},
                                       FastAI.OneHotTensor{0}))
    push!(variants,
          "backbone" => ModelVariant(make_cnn_backbone,
                                     ImageTensor{2},
                                     ConvFeatures{2}(nfeatures)))

    return variants
end

function make_cnn_classifier(model, input::ImageTensor, output::OneHotTensor{0})
    backbone = _backbone_with_channels(model.layers[1], input.nchannels)
    head = _head_with_classes(model.layers[2], length(output.classes))
    return Chain(backbone, head)
end

function make_cnn_classifier(model, ::Type{Any}, ::Type{Any})
    return model
end

function make_cnn_backbone(model, input::ImageTensor{N}, ::ConvFeatures{N}) where {N}
    backbone = _backbone_with_channels(model.layers[1], input.nchannels)
    return backbone
end

function make_cnn_backbone(model, ::Type{Any}, ::Type{Any})
    return model.layers[1]
end

function _backbone_with_channels(backbone, n)
    layer = backbone.layers[1].layers[1]
    layer isa Conv || throw(ArgumentError("""To change the number of input channels,
                                          `backbone.layers[1].layers[1]` must be a `Conv` layer."""))

    sz = size(layer.weight)
    ks = sz[begin:(end - 2)]
    in_, out = sz[(end - 1):end]
    in_ == n && return backbone

    layer = @set layer.weight = Flux.kaiming_normal(Random.GLOBAL_RNG, ks..., n, out)
    return @set backbone.layers[1].layers[1] = layer
end

function _head_with_classes(head, n)
    head.layers[end] isa Dense ||
        throw(ArgumentError("""To change the number of output classes,
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
        return modelfn(args...; pretrain = !isnothing(checkpoint), kwargs...)
    end
end

# model config: id, description, basefn, variant, hasweights, nfeatures
const METALHEAD_CONFIGS = [
    ("metalhead/resnet18", "ResNet18", metalhead_loadfn(Metalhead.ResNet, 18), true, 512),
    ("metalhead/resnet34", "ResNet34", metalhead_loadfn(Metalhead.ResNet, 34), true, 512),
    ("metalhead/resnet50", "ResNet50", metalhead_loadfn(Metalhead.ResNet, 50), true, 2048),
    ("metalhead/resnet101", "ResNet101", metalhead_loadfn(Metalhead.ResNet, 101), true,
     2048),
    ("metalhead/resnet152", "ResNet152", metalhead_loadfn(Metalhead.ResNet, 152), true,
     2048),
    ("metalhead/wideresnet50", "WideResNet50", metalhead_loadfn(Metalhead.WideResNet, 50),
     true, 2048),
    ("metalhead/wideresnet101", "WideResNet101",
     metalhead_loadfn(Metalhead.WideResNet, 101), true, 2048),
    ("metalhead/wideresnet152", "WideResNet152",
     metalhead_loadfn(Metalhead.WideResNet, 152), true, 2048),
    ("metalhead/googlenet", "GoogLeNet", metalhead_loadfn(Metalhead.GoogLeNet), false,
     1024),
    ("metalhead/inceptionv3", "InceptionV3", metalhead_loadfn(Metalhead.Inceptionv3), false,
     2048),
    ("metalhead/inceptionv4", "InceptionV4", metalhead_loadfn(Metalhead.Inceptionv4), false,
     1536),
    ("metalhead/squeezenet", "SqueezeNet", metalhead_loadfn(Metalhead.SqueezeNet), true,
     512),
    ("metalhead/densenet-121", "DenseNet121", metalhead_loadfn(Metalhead.DenseNet, 121),
     false, 1024),
    ("metalhead/densenet-161", "DenseNet161", metalhead_loadfn(Metalhead.DenseNet, 161),
     false, 1472),
    ("metalhead/densenet-169", "DenseNet169", metalhead_loadfn(Metalhead.DenseNet, 169),
     false, 1664),
    ("metalhead/densenet-201", "DenseNet201", metalhead_loadfn(Metalhead.DenseNet, 201),
     false, 1920),
    ("metalhead/resnext50", "ResNeXt50", metalhead_loadfn(Metalhead.ResNeXt, 50), true,
     2048),
    ("metalhead/resnext101", "ResNeXt101", metalhead_loadfn(Metalhead.ResNeXt, 101), true,
     2048),
    ("metalhead/resnext152", "ResNeXt152", metalhead_loadfn(Metalhead.ResNeXt, 152), true,
     2048),
    ("metalhead/mobilenetv1", "MobileNetV1", metalhead_loadfn(Metalhead.MobileNetv1), false,
     1024),
    ("metalhead/mobilenetv2", "MobileNetV2", metalhead_loadfn(Metalhead.MobileNetv2), false,
     1280),
    ("metalhead/mobilenetv3-small", "MobileNetV3 Small",
     metalhead_loadfn(Metalhead.MobileNetv3, :small), false, 576),
    ("metalhead/mobilenetv3-large", "MobileNetV3 Large",
     metalhead_loadfn(Metalhead.MobileNetv3, :large), false, 960),
    ("metalhead/efficientnet-b0", "EfficientNet-B0",
     metalhead_loadfn(Metalhead.EfficientNet, :b0), false, 1280),
    ("metalhead/efficientnet-b0", "EfficientNet-B0",
     metalhead_loadfn(Metalhead.EfficientNet, :b0), false, 1280),
    ("metalhead/efficientnet-b1", "EfficientNet-B1",
     metalhead_loadfn(Metalhead.EfficientNet, :b1), false, 1280),
    ("metalhead/efficientnet-b2", "EfficientNet-B2",
     metalhead_loadfn(Metalhead.EfficientNet, :b2), false, 1280),
    ("metalhead/efficientnet-b3", "EfficientNet-B3",
     metalhead_loadfn(Metalhead.EfficientNet, :b3), false, 1280),
    ("metalhead/efficientnet-b4", "EfficientNet-B4",
     metalhead_loadfn(Metalhead.EfficientNet, :b4), false, 1280),
    ("metalhead/efficientnet-b5", "EfficientNet-B5",
     metalhead_loadfn(Metalhead.EfficientNet, :b5), false, 1280),
    ("metalhead/efficientnet-b6", "EfficientNet-B6",
     metalhead_loadfn(Metalhead.EfficientNet, :b6), false, 1280),
    ("metalhead/efficientnet-b7", "EfficientNet-B7",
     metalhead_loadfn(Metalhead.EfficientNet, :b7), false, 1280),
    ("metalhead/efficientnet-b8", "EfficientNet-B8",
     metalhead_loadfn(Metalhead.EfficientNet, :b8), false, 1280),
]
for (id, description, loadfn, hasweights, nfeatures) in METALHEAD_CONFIGS
    _models[id] = (;
                   id, loadfn, description,
                   variants = cnn_variants(; hasweights, nfeatures),
                   checkpoints = hasweights ? ["imagenet1k"] : String[],
                   backend = :flux)
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

@testset "Metalhead models" begin for id in models(id = "metalhead")[:, :id]
    @test_nowarn load(models()[id]; variant = "backbone")
end end
