
# ## Model variants for Metalhead.jl models

struct MetalheadClassifierVariant <: ModelVariant
    fn
end
compatibleblocks(::MetalheadClassifierVariant) = (ImageTensor{2}, FastAI.OneHotTensor{0})
function loadvariant(v::MetalheadClassifierVariant, xblock::ImageTensor{2}, yblock::FastAI.OneHotTensor{0}, checkpoint; kwargs...)
    return v.fn(; pretrain = checkpoint == "imagenet1k", inchannels=xblock.nchannels,
                    nclasses=length(yblock.classes), kwargs...)
end
function loadvariant(v::MetalheadClassifierVariant, xblock, yblock, checkpoint; kwargs...)
    return v.fn(; pretrain = checkpoint == "imagenet1k", kwargs...)
end

struct MetalheadImageNetVariant <: ModelVariant
    fn
end
compatibleblocks(::MetalheadImageNetVariant) = (ImageTensor{2}(3), FastAI.OneHotTensor{0, Int}(1:1000))
function loadvariant(v::MetalheadImageNetVariant, xblock, yblock, checkpoint; kwargs...)
    return v.fn(; pretrain = checkpoint == "imagenet1k", kwargs...)
end

struct MetalheadBackboneVariant <: ModelVariant
    fn
    nfeatures::Int
end
compatibleblocks(variant::MetalheadBackboneVariant) = (ImageTensor{2}, ConvFeatures{2}(variant.nfeatures))
function loadvariant(v::MetalheadBackboneVariant, xblock::ImageTensor{2}, yblock::ConvFeatures{2}, checkpoint; kwargs...)
    model = v.fn(; pretrain = checkpoint == "imagenet1k", inchannels=xblock.nchannels,
                 kwargs...)
    return model.layers[1]
end
function loadvariant(v::MetalheadBackboneVariant, xblock, yblock, checkpoint; kwargs...)
    model = v.fn(; pretrain = checkpoint == "imagenet1k", kwargs...)
    return model.layers[1]
end

function metalheadvariants(modelfn, nfeatures)
    return [
        "imagenet1k" => MetalheadImageNetVariant(modelfn),
        "classifier" => MetalheadClassifierVariant(modelfn),
        "backbone" => MetalheadBackboneVariant(modelfn, nfeatures),
    ]
end


const _models = Dict{String, Any}()


fix(fn, args...; kwargs...) = (_args...; _kwargs...) -> fn(args..., _args...; kwargs..., _kwargs...)



# model config: id, description, basefn, variant, hasweights, nfeatures
const METALHEAD_CONFIGS = [
    ("metalhead/resnet18", "ResNet18", fix(Metalhead.ResNet, 18), true, 512),
    ("metalhead/resnet34", "ResNet34", fix(Metalhead.ResNet, 34), true, 512),
    ("metalhead/resnet50", "ResNet50", fix(Metalhead.ResNet, 50), true, 2048),
    ("metalhead/resnet101", "ResNet101", fix(Metalhead.ResNet, 101), true,
     2048),
    ("metalhead/resnet152", "ResNet152", fix(Metalhead.ResNet, 152), true,
     2048),
    ("metalhead/wideresnet50", "WideResNet50", fix(Metalhead.WideResNet, 50),
     true, 2048),
    ("metalhead/wideresnet101", "WideResNet101",
     fix(Metalhead.WideResNet, 101), true, 2048),
    ("metalhead/wideresnet152", "WideResNet152",
     fix(Metalhead.WideResNet, 152), true, 2048),
    ("metalhead/googlenet", "GoogLeNet", Metalhead.GoogLeNet, false,
     1024),
    ("metalhead/inceptionv3", "InceptionV3", Metalhead.Inceptionv3, false,
     2048),
    ("metalhead/inceptionv4", "InceptionV4", Metalhead.Inceptionv4, false,
     1536),
    ("metalhead/squeezenet", "SqueezeNet", Metalhead.SqueezeNet, true,
     512),
    ("metalhead/densenet-121", "DenseNet121", fix(Metalhead.DenseNet, 121),
     false, 1024),
    ("metalhead/densenet-161", "DenseNet161", fix(Metalhead.DenseNet, 161),
     false, 1472),
    ("metalhead/densenet-169", "DenseNet169", fix(Metalhead.DenseNet, 169),
     false, 1664),
    ("metalhead/densenet-201", "DenseNet201", fix(Metalhead.DenseNet, 201),
     false, 1920),
    ("metalhead/resnext50", "ResNeXt50", fix(Metalhead.ResNeXt, 50), true,
     2048),
    ("metalhead/resnext101", "ResNeXt101", fix(Metalhead.ResNeXt, 101), true,
     2048),
    ("metalhead/resnext152", "ResNeXt152", fix(Metalhead.ResNeXt, 152), true,
     2048),
    ("metalhead/mobilenetv1", "MobileNetV1", Metalhead.MobileNetv1, false,
     1024),
    ("metalhead/mobilenetv2", "MobileNetV2", Metalhead.MobileNetv2, false,
     1280),
    ("metalhead/mobilenetv3-small", "MobileNetV3 Small",
     fix(Metalhead.MobileNetv3, :small), false, 576),
    ("metalhead/mobilenetv3-large", "MobileNetV3 Large",
     fix(Metalhead.MobileNetv3, :large), false, 960),
    ("metalhead/efficientnet-b0", "EfficientNet-B0",
     fix(Metalhead.EfficientNet, :b0), false, 1280),
    ("metalhead/efficientnet-b0", "EfficientNet-B0",
     fix(Metalhead.EfficientNet, :b0), false, 1280),
    ("metalhead/efficientnet-b1", "EfficientNet-B1",
     fix(Metalhead.EfficientNet, :b1), false, 1280),
    ("metalhead/efficientnet-b2", "EfficientNet-B2",
     fix(Metalhead.EfficientNet, :b2), false, 1280),
    ("metalhead/efficientnet-b3", "EfficientNet-B3",
     fix(Metalhead.EfficientNet, :b3), false, 1280),
    ("metalhead/efficientnet-b4", "EfficientNet-B4",
     fix(Metalhead.EfficientNet, :b4), false, 1280),
    ("metalhead/efficientnet-b5", "EfficientNet-B5",
     fix(Metalhead.EfficientNet, :b5), false, 1280),
    ("metalhead/efficientnet-b6", "EfficientNet-B6",
     fix(Metalhead.EfficientNet, :b6), false, 1280),
    ("metalhead/efficientnet-b7", "EfficientNet-B7",
     fix(Metalhead.EfficientNet, :b7), false, 1280),
    ("metalhead/efficientnet-b8", "EfficientNet-B8",
     fix(Metalhead.EfficientNet, :b8), false, 1280),
]
for (id, description, loadfn, hasweights, nfeatures) in METALHEAD_CONFIGS
    _models[id] = (;
                   id, description,
                   variants = metalheadvariants(loadfn, nfeatures),
                   checkpoints = hasweights ? ["imagenet1k"] : String[],
                   backend = :flux)
end

@testset "Model variants" begin
    @testset "make_cnn_classifier" begin
        m = Metalhead.ResNet(18)
        clf = make_cnn_classifier(m, ImageTensor{2}(3), FastAI.OneHotTensor{0, Int}(1:10))
        @test Flux.outputsize(clf, (256, 256, 3, 1)) == (10, 1)

        clf2 = make_cnn_classifier(m, ImageTensor{2}(5), FastAI.OneHotTensor{0, Int}(1:100))
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
