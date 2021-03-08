function pixelshufflehead(k_in, k_out; n_upscale = 3, k_mid = 64, σ = relu)
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
