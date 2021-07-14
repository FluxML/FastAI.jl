function pixelshufflehead(k_in, k_out; n_upscale=3, k_mid=64, σ=relu)
    layers = []

    push!(layers, Chain(
        PixelShuffle(2, k_in, k_mid),
        BatchNorm(k_mid, σ),
    ))
    for i = 2:n_upscale
        push!(layers, Chain(
            PixelShuffle(2, k_mid),
            BatchNorm(k_mid, σ),

        ))
    end

    push!(layers, Conv((1, 1), k_mid => k_out))

    return Chain(layers...)
end


function visionhead(
        k_in,
        k_out;
        ks_dense=[512],
        p=0.,
        concat_pool=true,
        bn_first=true,
        bn_final=true,
        y_range=nothing,
        act=relu)

    hs = vcat([concat_pool ? 2k_in : k_in], ks_dense, [k_out])
    n = length(hs)
    bns = trues(n)
    bns[1] = bn_first
    acts = vcat([relu for _ ∈ 1:n-2], [identity])
    pool = concat_pool ? AdaptiveConcatPool((1, 1)) : AdaptiveMeanPool((1, 1))

    layers = [pool, flatten]

    for (h_in, h_out, act) in zip(hs, hs[2:end], acts)
        push!(layers, linbndrop(h_in, h_out, act=act, p=p))
    end

    if !isnothing(y_range)
        min, max = y_range
        push!(layers, x -> Flux.σ.(x) .* (max - min) .- max)
    end
    return Chain(layers...)
end


function linbndrop(h_in, h_out; use_bn=true, p=0., act=identity, lin_first=false)
    bn = BatchNorm(lin_first ? h_out : h_in)
    dropout = p == 0 ? identity : Dropout(p)
    dense = Dense(h_in, h_out, act; bias=!use_bn)
    if lin_first
        return Chain(dense, bn, dropout)
    else
        return Chain(bn, dropout, dense)
    end
end
