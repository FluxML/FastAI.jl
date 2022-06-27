
function createhandle(backend::ShowMakie; kwargs...)
    fig = Makie.Figure(; kwargs..., theme = Makie.theme_light(), backend.kwargs...)
    return fig
end

function _showblock!(grid::Makie.GridLayout, backend::ShowMakie, block::AbstractBlock, obs)
    ax = blockaxis(grid[1, 1], backend, block)
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
    label = Makie.Label(gp, title, textsize = 30, color = :gray40, lineheight = 1,
                        padding = (0, 0, 2, 0))
    label.tellwidth = false
    label.tellheight = true
end

function showblock(backend::ShowMakie, block, obs)
    # Calculate resolution based on number of blocks and size for one block
    fig = createhandle(backend)
    grid = fig[1, 1] = gridlayout()
    _showblock!(grid, backend, block, obs)

    Makie.resize!(fig, (_nblocks(block), 1) .* backend.size)
    Makie.resize_to_layout!(fig)
    return fig
end

function showblocks(backend::ShowMakie, block, obss::AbstractVector)
    # Calculate resolution based on number of blocks and size for one block
    fig = createhandle(backend)
    for (i, obs) in enumerate(obss)
        grid = fig[i, 1] = gridlayout()
        _showblock!(grid, backend, i == 1 ? block : _notitles(block), obs)
    end

    Makie.resize!(fig, (_nblocks(block), length(obss)) .* backend.size)
    Makie.resize_to_layout!(fig)
    return fig
end

function gridlayout()
    Makie.GridLayout(default_rowgap = 0.0, default_colgap = 16)
end

_nblocks(t::Tuple) = sum(_nblocks, t)
_nblocks(b::AbstractBlock) = 1
_nblocks((_, block)::Pair) = _nblocks(block)

_notitles(t::Tuple) = map(_notitles, t)
_notitles(b::AbstractBlock) = b
_notitles((_, block)::Pair) = _notitles(block)


@testset "ShowMakie" begin
    backend = ShowMakie()
    fig = createhandle(backend)

    block = Label(1:10)
    obs = 1
    @test_nowarn showblock(backend, block, obs)
    @test_nowarn showblock(backend, (block, block), (obs, obs))
    @test_nowarn showblock(backend, "Title" => block, obs)
    @test_nowarn showblock(backend, "Title" => ("Subtitle" => block, "Subtitle2" => block),
                           (obs, obs))

    @test_nowarn showblocks(backend, block, [obs, obs])
    @test_nowarn showblocks(backend, (block, block), [(obs, obs), (obs, obs)])
    @test_nowarn showblocks(backend, "Title" => block, [obs, obs])
    @test_nowarn showblocks(backend, "Title" => ("Subtitle" => block, "Subtitle2" => block),
                            [(obs, obs), (obs, obs)])
end
