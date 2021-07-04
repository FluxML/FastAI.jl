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

backbone = Metalhead.ResNet50(pretrain=true).layers[1:end-3]
unet = UNetDynamic(backbone, (256, 256, 3, 1); k_out = 10)
Flux.outputsize(unet, (256, 256, 3, 1)) == (256, 256, 10, 1)

unet = UNetDynamic(backbone, (256, 256, 3, 1); fdownscalk_out = 10)
Flux.outputsize(unet, (256, 256, 3, 1)) == (256, 256, 10, 1)
```
"""
function UNetDynamic(backbone, inputsize, final; kwargs...)
    backbonelayers = collect(iterlayers(backbone))
    unet = unet_from_layers(backbonelayers, inputsize; kwargs...)
    outsz = Flux.outputsize(unet, inputsize)
    return Chain(unet, final(outsz))
end

function UNetDynamic(backbone, inputsize, k_out::Int; kwargs...)
    final = insz -> convxlayer(insz[end-1], k_out; ks = 1)
    return UNetDynamic(backbone, inputsize, final; kwargs...)
end


function unet_from_layers(
        backbonelayers,
        insz;
        fdownscale = 0,
        upsample = upsample_block_small,
        agg = (mx, x) -> cat(mx, x; dims = length(insz)-1),
        kwargs...)
    layers = []
    channeldim = length(insz) - 1


    for (i, layer) in enumerate(backbonelayers)
        outsz = Flux.outputsize(layer, insz)
        if (insz[1] ÷ 2 == outsz[1])
            if fdownscale == 0
                child_unet = unet_from_layers(
                    backbonelayers[i+1:end],
                    outsz;
                    upsample = upsample,
                    fdownscale = fdownscale,
                    agg = agg)
                outsz = Flux.outputsize(child_unet, outsz)

                upsample_k_out = insz[channeldim]
                up = upsample(outsz, upsample_k_out; kwargs...)

                push!(layers, SkipConnection(Chain(layer, child_unet, up), agg))

                break
            else
                fdownscale -= 1
            end
        end
        push!(layers, layer)
        insz = outsz
    end

    unet = length(layers) == 1 ? only(layers) : Chain(layers...)
    return unet
end

iterlayers(m::Chain) = Iterators.flatten(iterlayers(l) for l in m.layers)
iterlayers(m) = (m,)

"""
    upsample_block_small(insize, k_out)

An upsampling block that increases the spatial dimensions of the input by 2
using pixel-shuffle upsampling.
"""
function upsample_block_small(insize, k_out; ks = 3, kwargs...)
    return Chain(
        Flux.PixelShuffle(2),
        convxlayer(insize[end-1] ÷ 4, k_out; kwargs...)
    )
end

function conv_final(insize, k_out; ks = 1, kwargs...)
    return convxlayer(insize[end-1], k_out; ks = ks, kwargs...)
end
