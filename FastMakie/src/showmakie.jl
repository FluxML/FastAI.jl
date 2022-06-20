

function createhandle(backend::ShowMakie; kwargs...)
    fig = Makie.Figure(; kwargs..., theme = Makie.theme_light(), backend.kwargs...)
    return fig
end


function _showblock!(grid::Makie.GridLayout, backend::ShowMakie, block::AbstractBlock, obs)
    ax = blockaxis(grid[1, 1], block)
    showblock!(ax, backend, block, obs)
    return ax
end


function _showblock!(grid::Makie.GridLayout, backend::ShowMakie, blocks::Tuple, obss)
    for (i, (block, obs)) in enumerate(zip(blocks, obss))
        subgrid = grid[1, i] = gridlayout()
        _showblock!(subgrid, backend, block, obs)
    end
end


function _showblock!(grid::Makie.GridLayout, backend::ShowMakie, (title, block)::Pair, obs)
    _addtitle(grid[1, 1], title)
    subgrid = grid[2, 1] = gridlayout()
    _showblock!(subgrid, backend, block, obs)
end

function _addtitle(gp::Makie.GridPosition, title)
    label = Makie.Label(gp, title, textsize=30, color=:gray40, lineheight=1, padding = (0, 0, 2, 0))
    label.tellwidth = false
    label.tellheight = true
end

function showblock(backend::ShowMakie, block, obs)
    # Calculate resolution based on number of blocks and size for one block
    fig = createhandle(backend)
    grid = fig[1, 1] = gridlayout()
    _showblock!(grid, backend, block, obs)

    Makie.resize!(fig, (_nblocks(block), 1)  .* backend.size)
    Makie.resize_to_layout!(fig)
    return fig
end

function showblocks(backend::ShowMakie, block, obss::AbstractVector)
    # Calculate resolution based on number of blocks and size for one block
    fig = createhandle(backend)
    for (i, obs) in enumerate(obss)
        grid = fig[i, 1] = gridlayout()
        _showblock!(grid, backend, i == 1 ? _withblockname(block) : _notitles(block), obs)
    end

    Makie.resize!(fig, (_nblocks(block), length(obss))  .* backend.size)
    Makie.resize_to_layout!(fig)
    return fig
end


function gridlayout()
    Makie.GridLayout(default_rowgap = 0., default_colgap = 16,)
end

_nblocks(t::Tuple) = sum(_nblocks, t)
_nblocks(b::AbstractBlock) = 1
_nblocks((_, block)::Pair) = _nblocks(block)


_notitles(t::Tuple) = map(_notitles, t)
_notitles(b::AbstractBlock) = b
_notitles((_, block)::Pair) = _notitles(block)

_withblockname(t::Tuple) = map(_withblockname, t)
_withblockname(b::AbstractBlock) = "($(nameof(typeof(b))))" => b
_withblockname((title, b)::Pair{String, <:AbstractBlock}) = "$title ($(nameof(typeof(b))))" => b
_withblockname((title, t)::Pair) = title => map(_withblockname, t)
#=

function showblock!(grid, backend::ShowMakie, blocks::Tuple, obss::Tuple)
    header = [block isa Pair ? first(block) : "" for block in blocks]
    blocks = Tuple(block isa Pair ? last(block) : block for block in blocks)

    Makie.rowsize!(grid, 1, Makie.Fixed(backend.size[2]))


    # Show blocks in a row
    col = 1
    for (i, (block, obs)) in enumerate(zip(blocks, obss))
        w = _nblocks(block)
        subgrid = grid[1, col:col+w-1] = Makie.GridLayout(tellheight = false)
        showblock!(subgrid, backend, block, obs)
        for j = col:col+w-1
            Makie.colsize!(grid, j, Makie.Fixed(backend.size[1]))
        end
        col += w
    end

    # Add titles to named blocks
    Makie.Label(grid[0, 1], "", tellwidth = false, textsize = 25)
    for (i, title) in enumerate(header)
        Makie.Label(grid[1, i], title, tellwidth = false, textsize = 25)
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
        subgrid = grid[i, 1:n] = Makie.GridLayout(tellheight = false)
        Makie.rowsize!(grid, i, Makie.Fixed(backend.size[2]))
        showblock!(subgrid, backend, blocks, obs)
    end

    for i = 1:n
        Makie.colsize!(grid, i, Makie.Fixed(backend.size[1]))
    end

    # Add titles to named blocks
    Makie.Label(grid[0, 1], "", textsize = 25)
    col = 1
    for (i, (title, block)) in enumerate(zip(header, blocks))
        w = _nblocks(block)
        Makie.Label(grid[1, col:col+w-1], title, tellwidth = false, textsize = 25)
        col += w
    end
end
showblocks!(grid, backend::ShowMakie, block, obss::AbstractVector) =
    showblocks!(grid, backend, (block,), map(obs -> (obs,), obss))

    =#
