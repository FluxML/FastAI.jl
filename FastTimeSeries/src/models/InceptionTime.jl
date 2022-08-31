"""
    InceptionModule(ni::Int, nf::Int, ks::Int = 40, bottleneck::Bool = true)

TBW
"""
function InceptionModule(ni::Int, nf::Int, ks::Int = 40, bottleneck::Bool = true)
    ks = [ks ÷ (2^i) for i in range(0, stop = 2)]
    ks = [ks[i] % 2 == 0 ? ks[i] - 1 : ks[i] for i in range(1, stop = 3)]  # ensure odd ks
    bottleneck = ni > 1 ? bottleneck : false

    bottleneck_block = bottleneck ? Conv1d(ni, nf, 1, bias = false) : identity

    convs_layers =
        [Conv1d(bottleneck ? nf : ni, nf, ks[i], bias = false) for i in range(1, stop = 3)]

    convs = Chain(bottleneck_block, Parallel(hcat, convs_layers...))

    maxconvpool = Chain(MaxPool((3,), pad = 1, stride = 1), Conv1d(ni, nf, 1, bias = false))

    return Chain(Parallel(hcat, convs, maxconvpool), BatchNorm(nf * 4, relu))
end

"""
    InceptionBlock(ni::Int, nf::Int = 32, residual::Bool = true, depth::Int = 6)

TBW
"""
function InceptionBlock(ni::Int, nf::Int = 32, residual::Bool = true, depth::Int = 6)
    inception = []
    shortcut = []

    for d in range(1, stop = depth)
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

function changedims(X)
    X = permutedims(X, (2, 1, 3))
    return X
end

"""
    InceptionTime(c_in::Int, c_out::Int, seq_len = nothing, nf::Int = 32)

TBW
"""
function InceptionTime(c_in::Int, c_out::Int, seq_len = nothing, nf::Int = 32)
    inceptionblock = InceptionBlock(c_in, nf)
    gap = GAP1d(1)
    fc = Dense(nf * 4, c_out)
    return Chain(changedims, inceptionblock, gap, fc)
end
