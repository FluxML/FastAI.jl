
act_fn = relu

convx(ni, nf; ks=3, stride=1) = Conv(
        (ks, ks), ni => nf, stride = stride, pad = ks ÷ 2, init = Flux.kaiming_normal)

function convxlayer(ni, nf; ks=3, stride=1, zero_bn=false, act=true)
    bn = BatchNorm(nf, act ? act_fn : identity)
    fill!(bn.γ, zero_bn ? 0 : 1)
    layers = [convx(ni, nf; ks = ks, stride = stride), bn]
    return Chain(layers...)
end

struct ResBlock
    convs
    idconv
    pool
end

function ResBlock(expansion::Int, ni::Int, nh::Int; stride::Int = 1)
    nf, ni = nh * expansion, ni * expansion
    if expansion == 1
        layers = [
            convxlayer(ni, nh; stride = stride),
            convxlayer(nh, nf; zero_bn = true, act = false),
        ]
    else
        layers = [
            convxlayer(ni, nh; ks = 1),
            convxlayer(nh, nh; stride = stride),
            convxlayer(nh, nf; ks = 1, zero_bn = true, act = false)
        ]
    end

    return ResBlock(
        Chain(layers...),
        ni == nf ? identity : convxlayer(ni, nf; ks = 1, stride = 1),
        stride == 1 ? identity : MeanPool((2, 2))
    )
end
Flux.@functor ResBlock


(r::ResBlock)(x) = act_fn.(r.convs(x) .+ r.idconv(r.pool(x)))

function make_layer(expansion, ni, nf, n_blocks, stride)
    return Chain([
            ResBlock(expansion, i == 1 ? ni : nf, nf; stride =(i == 1 ? stride : 1))
            for i in 1:n_blocks
            ]...)
end

function XResNet(expansion, layers; c_in = 3)
    nfs = [c_in, (c_in+1)*8, 64, 64]
    stem = [convxlayer(nfs[i], nfs[i+1]; stride = i == 1 ? 2 : 1) for i in 1:3]

    nfs = [64 ÷ expansion, 64, 128, 256, 512]
    res_layers = [
        make_layer(expansion, nfs[i], nfs[i+1], l, i == 1 ? 1 : 2)
        for (i, l) in enumerate(layers)
    ]

    return Chain(
        stem...,
        MaxPool((3, 3); pad = 1, stride = 2),
        res_layers...,
    )
end

xresnet18(;init = true, kwargs...) = XResNet(1, [2, 2, 2, 2]; kwargs...)
xresnet50(;init = true, kwargs...) = XResNet(4, [3, 4, 6, 3]; kwargs...)
