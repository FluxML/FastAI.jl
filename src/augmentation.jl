"""
    augs_projection([; kwargs...])

Helper to create a set of projective transformations for image, mask
and keypoint data. Similar to fastai's
[`aug_transforms`](https://github.com/fastai/fastai/blob/bdc58846753c6938c63344fcaebea7149585fd5c/fastai/vision/augment.py#L946).

## Keyword arguments

- `flipx = true`: Whether to perform a horizontal flip with probability `1/2`. See [`FlipX`](#).
- `flipy = false`: Whether to perform a vertical flip with probability `1/2`. See [`FlipY`](#).
- `max_zoom = 1.5`: Maximum factor by which to zoom. Set to `1.` to disable. See [`Zoom`](#).
- `max_rotate = 10`: Maximum absolute degree by which to rotate. Set to `0.` to disable. See [`Rotate`](#).
- `max_warp = 0.05`: Intensity of corner warp. Set to `0.` to disable. See [`WarpAffine`](#).
"""
function augs_projection(;
        flipx=true,
        flipy=false,
        max_zoom=1.5,
        max_rotate=10.,
        max_warp=0.05,
        )
    tfms = []

    flipx && push!(tfms, Maybe(FlipX()))
    flipy && push!(tfms, Maybe(FlipY()))
    max_warp > 0 && push!(tfms, WarpAffine(max_warp))
    max_rotate > 0 && push!(tfms, Rotate(max_rotate))
    push!(tfms, Zoom((1., max_zoom)))
    return DataAugmentation.compose(tfms...)
end

"""
    augs_lighting([; intensity = 0.2, p = 0.75])

Helper to create a set of lighting transformations for image data. With
probability `p`, applies [`AdjustBrightness`](#)`(intensity)` and
[`AdjustContrast`](#)`(intensity)`.
"""
function augs_lighting(;intensity=0.2, p=0.75)
    return Maybe(AdjustBrightness(intensity), p) |> Maybe(AdjustContrast(intensity), p)
end
