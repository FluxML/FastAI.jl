"""
    InceptionModule(ni::Int, nf::Int, ks::Int = 40, bottleneck::Bool = true)

An InceptionModule consists of an (optional) bottleneck, followed by
3 conv1d layers.
"""
function InceptionModule(ni::Int, nf::Int, kernel_size::Int = 40, bottleneck::Bool = true)
    ks = [kernel_size ÷ (2^i) for i in 0:2]
    ks = [ks[i] % 2 == 0 ? ks[i] - 1 : ks[i] for i in 1:3]  # ensure odd ks
    bottleneck = ni > 1 ? bottleneck : false

    bottleneck_block = bottleneck ? Conv1d(ni, nf, 1, bias = false) : identity

    convs_layers =
        [Conv1d(bottleneck ? nf : ni, nf, ks[i], bias = false) for i in 1:3]

    convs = Chain(bottleneck_block, Parallel(hcat, convs_layers...))

    maxconvpool = Chain(MaxPool((3,), pad = 1, stride = 1), Conv1d(ni, nf, 1, bias = false))

    return Chain(Parallel(hcat, convs, maxconvpool), BatchNorm(nf * 4, relu))
end

"""
    InceptionBlock(ni::Int, nf::Int = 32, residual::Bool = true, depth::Int = 6)

An InceptionBlock consists of variable number of InceptionModule depending on the depth.
Optionally residual.
"""
function InceptionBlock(ni::Int, nf::Int = 32, residual::Bool = true, depth::Int = 6)
    inception = []
    shortcut = []

    for d in 1:depth
        push!(inception, InceptionModule(d == 1 ? ni : nf * 4, nf))
        if residual && d % 3 == 0
            n_in = d == 3 ? ni : nf * 4
            n_out = nf * 4
            skip =
                n_in == n_out ? BatchNorm(n_out) :
                Chain(Conv1d(n_in, n_out, 1), BatchNorm(n_out))
            push!(shortcut, skip)
        end
    end

    blocks = []
    d = 1

    while d <= depth
        blk = []
        while d <= depth
            push!(blk, inception[d])
            if d % 3 == 0
                d += 1
                break
            end
            d += 1
        end
        if residual && d ÷ 3 <= length(shortcut)
            skp = shortcut[d÷3]
            push!(blocks, Parallel(+, Chain(blk...), skp))
        else
            push!(blocks, Chain(blk...))
        end
    end
    return Chain(blocks...)
end

changedims(X) = permutedims(X, (2, 1, 3))

"""
    InceptionTime(c_in::Int, c_out::Int, seq_len = nothing, nf::Int = 32)

A Julia Implemention of the InceptionTime model.
From https://arxiv.org/abs/1909.04939

## Arguments.

- `c_in` : The number of input channels.
- `c_out`: The number of output classes.
- `nf`   : The number of "hidden channels" to use.
"""
function InceptionTime(c_in::Int, c_out::Int, nf::Int = 32)
    inceptionblock = InceptionBlock(c_in, nf)
    gap = GAP1d(1)
    fc = Dense(nf * 4, c_out)
    return Chain(changedims, inceptionblock, gap, fc)
end
