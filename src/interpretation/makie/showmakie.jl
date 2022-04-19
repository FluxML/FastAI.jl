
using FastAI
import FastAI: createhandle, showblock!


function createhandle(backend::ShowMakie; kwargs...)
    fig = M.Figure(; kwargs..., backend.kwargs...)
    grid = fig[1, 1] = M.GridLayout()
    return grid
end

function showblock(backend::ShowMakie, block, obs)
    # Calculate resolution based on number of blocks and size for one block
    # A margin of 10% is added for titles
    width = _nblocks(block) * backend.size[2]
    height = round(Int, 1.1 * backend.size[1])


    grid = createhandle(backend, resolution = (width, height))
    fig = grid.parent.parent
    showblock!(grid, backend, block, obs)
    return fig
end

_nblocks(t::Tuple) = sum(_nblocks, t)
_nblocks(b::AbstractBlock) = 1
_nblocks((_, block)::Pair) = _nblocks(block)


function showblock!(grid, backend::ShowMakie, (title, block)::Pair, obs)
    showblock!(grid[1, 1], backend, block, obs)
end


function showblock!(grid, backend::ShowMakie, blocks::Tuple, obss::Tuple)
    header = [block isa Pair ? first(block) : "" for block in blocks]
    blocks = Tuple(block isa Pair ? last(block) : block for block in blocks)

    M.rowsize!(grid, 1, M.Fixed(backend.size[2]))


    # Show blocks in a row
    col = 1
    for (i, (block, obs)) in enumerate(zip(blocks, obss))
        w = _nblocks(block)
        subgrid = grid[1, col:col+w-1] = M.GridLayout(tellheight = false)
        showblock!(subgrid, backend, block, obs)
        for j = col:col+w-1
            M.colsize!(grid, j, M.Fixed(backend.size[1]))
        end
        col += w
    end

    # Add titles to named blocks
    M.Label(grid[0, 1], "", tellwidth = false, textsize = 25)
    for (i, title) in enumerate(header)
        M.Label(grid[1, i], title, tellwidth = false, textsize = 25)
    end

end


function showblocks(backend::ShowMakie, block, obss)
    width = _nblocks(block) * backend.size[2]
    height = round(Int, length(obss) * 1.1 * backend.size[1])
    res = (width * 1.2, height * 1.1)

    grid = createhandle(backend, resolution = res)
    fig = grid.parent.parent

    showblocks!(grid, backend, block, obss)
    return fig
end


function showblocks!(grid, backend::ShowMakie, blocks::Tuple, obss::AbstractVector)
    header = [block isa Pair ? first(block) : "" for block in blocks]
    blocks = Tuple(block isa Pair ? last(block) : block for block in blocks)
    n = _nblocks(blocks)


    # Show each sample in one row
    for (i, obs) in enumerate(obss)
        subgrid = grid[i, 1:n] = M.GridLayout(tellheight = false)
        M.rowsize!(grid, i, M.Fixed(backend.size[2]))
        showblock!(subgrid, backend, blocks, obs)
    end

    for i = 1:n
        M.colsize!(grid, i, M.Fixed(backend.size[1]))
    end

    # Add titles to named blocks
    M.Label(grid[0, 1], "", textsize = 25)
    col = 1
    for (i, (title, block)) in enumerate(zip(header, blocks))
        w = _nblocks(block)
        M.Label(grid[1, col:col+w-1], title, tellwidth = false, textsize = 25)
        col += w
    end
end
showblocks!(grid, backend::ShowMakie, block, obss::AbstractVector) =
    showblocks!(grid, backend, (block,), map(obs -> (obs,), obss))

## Block definitions


function showblock!(grid, ::ShowMakie, block::Label, obs)
    ax = cleanaxis(grid[1, 1])
    M.text!(ax, string(obs), space = :data)
end

function showblock!(grid, ::ShowMakie, block::LabelMulti, obs)
    ax = cleanaxis(grid[1, 1])
    M.text!(ax, join(string.(obs), "\n"), space = :data)
end


function showblock!(
    grid,
    ::ShowMakie,
    block::Union{<:OneHotTensor{0},<:OneHotTensorMulti{0}},
    obs,
)
    if !(sum(obs) ≈ 1)
        obs = softmax(obs)
    end
    ax = M.Axis(grid[1, 1], yticks = (1:length(block.classes), string.(block.classes)))
    M.barplot!(ax, obs, direction = :x)
    M.hidespines!(ax)
end


function default_showbackend()
    if ismissing(M.current_backend[])
        return ShowText()
    else
        return ShowMakie()
    end
end


## Helpers

"""
    cleanaxis(f)

Create a `Makie.Axis` with no interactivity, decorations and aspect distortion.
"""
function cleanaxis(f; kwargs...)
    ax = M.Axis(f; kwargs...)
    ax.aspect = M.DataAspect()
    ax.xzoomlock = true
    ax.yzoomlock = true
    ax.xrectzoom = false
    ax.yrectzoom = false
    ax.panbutton = nothing
    ax.xpanlock = true
    ax.ypanlock = true
    ax.bottomspinevisible = false
    ax.leftspinevisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    M.hidedecorations!(ax)

    return ax
end
