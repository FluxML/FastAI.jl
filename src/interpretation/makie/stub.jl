
"""
    ShowMakie([; kwargs...]) <: ShowBackend

A backend for showing block data that uses
[Makie.jl](https://github.com/JuliaPlots/Makie.jl) figures for
visualization.

Keyword arguments are passed through to the constructed `Figure`s.
"""
struct ShowMakie <: ShowBackend
    size::Tuple{Int,Int}
    kwargs::Any
end
ShowMakie(sz = (500, 500); kwargs...) = ShowMakie(sz, kwargs)
