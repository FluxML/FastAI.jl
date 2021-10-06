

"""
    ShowMakie([; kwargs...]) <: ShowBackend

A backend for showing block data that uses
[Makie.jl](https://github.com/JuliaPlots/Makie.jl) figures for
visualization.

Keyword arguments are passed through to the constructed `Figure`s.
"""
struct ShowMakie <: ShowBackend
    size::Tuple{Int,Int}
    kwargs
end
ShowMakie(sz=(500, 500); kwargs...) = ShowMakie(sz, kwargs)


function createhandle(backend::ShowMakie; kwargs...)
    fig = Figure(;kwargs..., backend.kwargs...)
    grid = fig[1, 1] = GridLayout()
    return grid
end

function showblock(backend::ShowMakie, block, data)
    # Calculate resolution based on number of blocks and size for one block
    # A margin of 10% is added for titles
    width = _nblocks(block) * backend.size[2]
    height = round(Int, 1.1 * backend.size[1])
    res = (width, height)

    grid = createhandle(backend, resolution=res)
    fig = grid.parent.parent
    showblock!(grid, backend, block, data)
    return fig
end

_nblocks(t::Tuple) = sum(_nblocks, t)
_nblocks(b::AbstractBlock) = 1
_nblocks((_, block)::Pair) = _nblocks(block)


function showblock!(grid, backend::ShowMakie, (title, block)::Pair, data)
    showblock!(grid[1, 1], backend, block, data)
    # title = Makie.Label(grid[0, 1], title, tellwidth=false, textsize=25)

end


function showblock!(grid, backend::ShowMakie, blocks::Tuple, datas::Tuple)
    header = [block isa Pair ? first(block) : "" for block in blocks]
    blocks = Tuple(block isa Pair ? last(block) : block for block in blocks)

    rowsize!(grid, 1, Makie.Fixed(backend.size[2]))


    # Show blocks in a row
    col = 1
    for (i, (block, data)) in enumerate(zip(blocks, datas))
        w = _nblocks(block)
        subgrid = grid[1, col:col+w-1] = GridLayout(tellheight=false)
        showblock!(subgrid, backend, block, data)
        for j in col:col+w-1
            colsize!(grid, j, Makie.Fixed(backend.size[1]))
        end
        col += w
    end

    # Add titles to named blocks
    Makie.Label(grid[0, 1], "", tellwidth=false, textsize=25)
    for (i, title) in enumerate(header)
        Makie.Label(grid[1, i], title, tellwidth=false, textsize=25)
    end

end


function showblocks(backend::ShowMakie, block, datas)
    width = _nblocks(block) * backend.size[2]
    height = round(Int, length(datas) * backend.size[1] + 0.2 * backend.size[1])
    res = (width, height)

    grid = createhandle(backend, resolution=res)
    fig = grid.parent.parent

    showblocks!(grid, backend, block, datas)
    return fig
end


function showblocks!(grid, backend::ShowMakie, blocks::Tuple, datas::AbstractVector)
    header = [block isa Pair ? first(block) : "" for block in blocks]
    blocks = Tuple(block isa Pair ? last(block) : block for block in blocks)
    n = _nblocks(blocks)


    # Show each sample in one row
    for (i, data) in enumerate(datas)
        subgrid = grid[i, 1:n] = GridLayout(tellheight=false)
        rowsize!(grid, i, Makie.Fixed(backend.size[2]))
        showblock!(subgrid, backend, blocks, data)
    end

    for i in 1:n
        colsize!(grid, i, Makie.Fixed(backend.size[1]))
    end

    # Add titles to named blocks
    Makie.Label(grid[0, 1], "", textsize=25)
    col = 1
    for (i, (title, block)) in enumerate(zip(header, blocks))
        w = _nblocks(block)
        Makie.Label(grid[1, col:col+w-1], title, tellwidth=false, textsize=25)
        col += w
    end
end
showblocks!(grid, backend::ShowMakie, block, datas::AbstractVector) =
    showblocks!(grid, backend, (block,), map(data -> (data,), datas))

## Block definitions


function showblock!(grid, ::ShowMakie, block::Label, data)
    ax = imageaxis(grid[1, 1])
    text!(ax, string(data), space=:data)
end

function showblock!(grid, ::ShowMakie, block::LabelMulti, data)
    ax = imageaxis(grid[1, 1])
    text!(ax, join(string.(data), "\n"), space=:data)
end


function showblock!(grid, ::ShowMakie, block::Union{<:OneHotTensor{0},<:OneHotTensorMulti{0}}, data)
    ax = Axis(grid[1, 1], yticks=(1:length(block.classes), string.(block.classes),))
    barplot!(
		ax,
		data,
		direction=:x,
	)
    hidespines!(ax)
end


function showblock!(grid, ::ShowMakie, block::Image{2}, data)
    ax = imageaxis(grid[1, 1])
    plotimage!(ax, data)
end
function showblock!(grid, ::ShowMakie, block::Mask{2}, data)
    ax = imageaxis(grid[1, 1])
    plotmask!(ax, data, block.classes)
end


function showblock!(grid, ::ShowMakie, block::Keypoints{2}, data)
    ax = imageaxis(grid[1, 1])
    h = maximum(first.(data))
    ks = [SVector(x, h-y) for (y, x) in data]
    scatter!(ax, ks)
end

function showblock!(grid, ::ShowMakie, block::Bounded{2, <:Keypoints{2}}, data)
    ax = imageaxis(grid[1, 1])
    h, w = block.size
    ks = [SVector(x, h-y) for (y, x) in data]
    xlims!(ax, 0, w)
    ylims!(ax, 0, h)
    scatter!(ax, ks)
end
