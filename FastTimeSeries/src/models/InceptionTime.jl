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

struct InceptionBlock
    depth::Int
    inception::Any
    shortcut::Any
end

function InceptionBlock(ni::Int, nf::Int = 32, residual::Bool = true, depth::Int = 6)
    inception = []
    #TODO: Add The Conv Block Layer and then finish the residual part.
    shortcut = []
    for d in range(0, stop = depth - 1)
        push!(inception, InceptionModule(d == 0 ? ni : nf * 4, nf))
        if (residual && d % 3 == 2)
            n_in = d == 2 ? ni : nf * 4
            n_out = nf * 4
        end
    end
    return InceptionBlock(depth, inception, shortcut)
end
Flux.@functor InceptionBlock
Flux.trainable(m::InceptionBlock) = (m.inception, m.shortcut)

# Model Output
function (m::InceptionBlock)(x)
    #TODO: Add The Conv Block Layer and then finish the residual part.
    res = x
    for d in range(1, stop = m.depth)
        x = m.inception[d](x)
    end
    return x
end

function InceptionTime(c_in::Int, c_out::Int, seq_len = nothing, nf::Int = 32)
    inceptionblock = InceptionBlock(c_in, nf, false)
    gap = GAP1d(1)
    fc = Dense(nf * 4, c_out)
    return Chain(inceptionblock, gap, fc)
end
