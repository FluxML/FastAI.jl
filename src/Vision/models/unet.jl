"""
    UNetDynamic(backbone, inputsize[; kwargs...])

Create a U-Net model from convolutional `backbone` architecture. After every
downsampling layer (i.e. pooling or strided convolution), a skip connection and
an upsampling block are inserted, resulting in a convolutional network with
the same spatial output dimensions as its input.

## Keyword arguments

- `upsample`: A *constructor* for an upsampling block callable with `upsample(insize, k_out)`.
    If `insize` is `(h, w, k, b)`, then the output should have size `(2h, 2w, k_out, b)`.
    Defaults to [`FastAI.Models.upsample_block_small`](#).
- `agg`: Aggregation function for skip connection. Default concatenates in the
  channel dimension. Use `+` for summing and see [`Flux.SkipConnection`](#) for more
  details.
- `fdownsample = 0`: Number of upsampling steps to leave out. By default there will be one
    upsampling step for every downsampling step in `backbone`. Hence if the input spatial
    size is `(h, w)`, the output size will be `(h/2^fdownsample, w/2^fdownsample)`, i.e.
    to get outputs at half the resolution, set `fdownsample = 1`.
- `kwargs...`: Other keyword arguments are passed through to `upsample`.

## Examples

```julia
using FastAI, Metalhead

backbone = Metalhead.ResNet50(pretrain=true).layers[1][1:end-1]
unet = UNetDynamic(backbone, (256, 256, 3, 1); k_out = 10)
Flux.outputsize(unet, (256, 256, 3, 1)) == (256, 256, 10, 1)

unet = UNetDynamic(backbone, (256, 256, 3, 1); fdownscalk_out = 10)
Flux.outputsize(unet, (256, 256, 3, 1)) == (256, 256, 10, 1)
```
"""
function UNetDynamic(
    backbone,
    inputsize,
    k_out::Int;
    final = UNetFinalBlock,
    fdownscale = 0,
    kwargs...,
)
    backbonelayers = collect(iterlayers(backbone))
    unet = unetlayers(
        backbonelayers,
        inputsize;
        m_middle = UNetMiddleBlock,
        skip_upscale = fdownscale,
        kwargs...,
    )
    outsz = Flux.outputsize(unet, inputsize)
    return Chain(unet, final(outsz[end-1], k_out))
end


function catchannels(x1, x2)
    ndims(x1) == ndims(x2) || error("Expected inputs with same number of dimensions!")
    cat(x1, x2; dims = ndims(x1) - 1)
end


function unetlayers(
    layers,
    sz;
    k_out = nothing,
    skip_upscale = 0,
    m_middle = _ -> (identity,),
)
    isempty(layers) && return m_middle(sz[end-1])

    layer, layers = layers[1], layers[2:end]
    outsz = Flux.outputsize(layer, sz)
    does_downscale = sz[1] ÷ 2 == outsz[1]

    if !does_downscale
        # If `layer` does not scale down the spatial dimensions, append
        # it to a Chain
        return Chain(layer, unetlayers(layers, outsz; k_out, skip_upscale)...)
    elseif does_downscale && skip_upscale > 0
        # If `layer` does scale down the spatial dimensions, but we don't
        # to upsample this one, recurse with modified arguments
        return Chain(
            layer,
            unetlayers(layers, outsz; skip_upscale = skip_upscale - 1, k_out)...,
        )
    else
        # `layer` scales down the spatial dimensions and we add an upsampling block
        # and a skip connection that scales the dimensions back up
        childunet = Chain(unetlayers(layers, outsz; skip_upscale)...)
        outsz = Flux.outputsize(childunet, outsz)

        k_in = sz[end-1]
        k_mid = outsz[end-1]
        k_out = isnothing(k_out) ? k_in : k_out
        return FastAI.Vision.Models.UNetBlock(
            Chain(layer, childunet),
            k_in,  # Input channels to upsampling layer
            k_mid,
            k_out,
        )
    end
end

iterlayers(m::Chain) = Iterators.flatten(iterlayers(l) for l in m.layers)
iterlayers(m) = (m,)


"""
    UNetBlock(m, k_in)

Given convolutional module `m` that halves the spatial dimensions
and outputs `k_in` filters, create a module that upsamples the
spatial dimensions and then aggregates features via  a skip connection.
"""
function UNetBlock(m_child, k_in, k_mid, k_out = 2k_in)
    return Chain(
        upsample = SkipConnection(
            Chain(
                child = m_child,                              # Downsampling and processing
                upsample = PixelShuffleICNR(k_mid, k_mid),  # Upsampling
            ),
            Parallel(catchannels, identity, BatchNorm(k_in)),
        ),
        act = xs -> relu.(xs),
        combine = UNetCombineLayer(k_in + k_mid, k_out),  # Data from both branches is combined
    )
end


function PixelShuffleICNR(k_in, k_out; r = 2)
    return Chain(convxlayer(k_in, k_out * (r^2), ks = 1), Flux.PixelShuffle(r))
end


function UNetCombineLayer(k_in, k_out)
    return Chain(convxlayer(k_in, k_out), convxlayer(k_out, k_out))
end

function UNetMiddleBlock(k)
    return Chain(convxlayer(k, 2k), convxlayer(2k, k))
end


function UNetFinalBlock(k_in, k_out)
    return Chain(ResBlock(1, k_in, k_in), convxlayer(k_in, k_out, ks = 1))
end

"""
    upsample_block_small(insize, k_out)

An upsampling block that increases the spatial dimensions of the input by 2
using pixel-shuffle upsampling.
"""
function upsample_block_small(insize, k_out; ks = 3, kwargs...)
    return Chain(Flux.PixelShuffle(2), convxlayer(insize[end-1] ÷ 4, k_out; kwargs...))
end

function conv_final(insize, k_out; ks = 1, kwargs...)
    return convxlayer(insize[end-1], k_out; ks = ks, kwargs...)
end
