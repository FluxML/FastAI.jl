
const act_fn = relu

function convx(ni, nf; ks = 3, stride = 1, ndim = 2)
    Conv(ntuple(_ -> ks, ndim),
         ni => nf,
         stride = stride,
         pad = ks ÷ 2,
         init = Flux.kaiming_normal)
end

function convxlayer(ni, nf; ks = 3, stride = 1, zero_bn = false, act = true, ndim = 2)
    bn = BatchNorm(nf, act ? act_fn : identity)
    fill!(bn.γ, zero_bn ? 0 : 1)
    return Chain(convx(ni, nf; ks = ks, stride = stride, ndim = ndim), bn)
end

struct ResBlock
    convs::Any
    idconv::Any
    pool::Any
end

function ResBlock(expansion::Int, ni::Int, nh::Int; stride::Int = 1, ndim = 2)
    nf, ni = nh * expansion, ni * expansion
    if expansion == 1
        layers = [
            convxlayer(ni, nh; stride = stride, ndim = ndim),
            convxlayer(nh, nf; zero_bn = true, act = false, ndim = ndim),
        ]
    else
        layers = [
            convxlayer(ni, nh; ks = 1, ndim = ndim),
            convxlayer(nh, nh; stride = stride, ndim = ndim),
            convxlayer(nh, nf; ks = 1, zero_bn = true, act = false, ndim = ndim),
        ]
    end

    return ResBlock(Chain(layers...),
                    ni == nf ? identity :
                    convxlayer(ni, nf; ks = 1, stride = 1, ndim = ndim),
                    stride == 1 ? identity : MeanPool(ntuple(_ -> 2, ndim)))
end
Flux.@functor ResBlock

(r::ResBlock)(x) = act_fn.(r.convs(x) .+ r.idconv(r.pool(x)))

function make_layer(expansion, ni, nf, n_blocks, stride; ndim = 2)
    return Chain([ResBlock(expansion,
                           i == 1 ? ni : nf,
                           nf;
                           stride = (i == 1 ? stride : 1),
                           ndim = ndim) for i in 1:n_blocks]...)
end

"""
    XResNet(expansion, layers; c_in = 3, ndim = 2)

Create an XResNet model backbone following the [implementation in
fastai](https://github.com/fastai/fastai/blob/master/fastai/vision/models/xresnet.py#L22).

- `c_in::Int = 3`: The number of input channels, e.g. `1` for grayscale images and `3` for
    RGB images
- `ndim::Int = 2`: The number of dimensions for the convolutional and pooling layers, e.g.
    `2` for 2D input images and `3` for 3D volumes.
"""
function XResNet(expansion, layers; c_in = 3, ndim = 2)
    nfs = [c_in, (c_in + 1) * 8, 64, 64]
    stem = [convxlayer(nfs[i], nfs[i + 1]; stride = i == 1 ? 2 : 1, ndim = ndim)
            for i in 1:3]

    nfs = [64 ÷ expansion, 64, 128, 256, 512]
    res_layers = [make_layer(expansion, nfs[i], nfs[i + 1], l, i == 1 ? 1 : 2, ndim = ndim)
                  for
                  (i, l) in enumerate(layers)]

    return Chain(stem..., MaxPool(ntuple(_ -> 3, ndim); pad = 1, stride = 2), res_layers...)
end

xresnet18(; kwargs...) = XResNet(1, [2, 2, 2, 2]; kwargs...)
xresnet50(; kwargs...) = XResNet(4, [3, 4, 6, 3]; kwargs...)

@testset "XResNet [model]" begin
    @testset "Basic" begin @test_nowarn begin
        model = xresnet18()
        @test Flux.outputsize(model, (128, 128, 3, 1)) == (4, 4, 512, 1)
    end end

    @testset "3D" begin @test_nowarn begin
        model = xresnet18(ndim = 3)
        @test Flux.outputsize(model, (128, 128, 128, 3, 1)) == (4, 4, 4, 512, 1)
    end end
end
