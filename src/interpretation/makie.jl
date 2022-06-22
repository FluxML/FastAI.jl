
"""
    ShowMakie([; kwargs...]) <: ShowBackend

A backend for showing block data that uses
[Makie.jl](https://github.com/JuliaPlots/Makie.jl) figures for visualization.

Keyword arguments `kwargs` are passed to the constructed `Figure`s.

## Implementing a `Block` visualization

As with other [`ShowBackend`](#s), implementing a visualization for a
block type `B <: AbstractBlock` requires you to implement [`showblock!`](#).

For `ShowMakie`, the first argument is a `Makie.Axis`, i.e. you have to
implement

```julia
FastAI.showblock!(ax::Makie.Axis, ::ShowMakie, block::B, obs)
```

The axis is created by [`FastMakie.makeaxis`](#). The default options will result in
an axis cleaned of all decorations. To customize it, implement
`FastMakie.axiskwargs(block::B)`. See the docstring of [`makeaxis`](#) for available options.
"""
struct ShowMakie <: ShowBackend
    size::Tuple{Int, Int}
    kwargs::Any
end
ShowMakie(sz = (500, 500); kwargs...) = ShowMakie(sz, kwargs)

axiskwargs(::AbstractBlock) = (;)
