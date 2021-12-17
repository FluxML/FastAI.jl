

"""
    ShowText([io; kwargs...]) <: ShowBackend

A backend for showing block data using text for REPL use.
Text is displayed to `io` and `kwargs` are keyword arguments
for `PrettyTables.pretty_table`, which is used to display
collections of blocks.
"""
struct ShowText <: ShowBackend
    io
    kwargs
end

ShowText(io=stdout; hlines=:all, alignment=:l, kwargs...) = ShowText(
    io,
    (; hlines=hlines, alignment=alignment, kwargs...))

createhandle(backend::ShowText) = backend.io


function showblock!(io, backend::ShowText, (title, block)::Pair, obs)
    printstyled(io, title, bold=true)
    println(io)
    showblock!(io, backend, block, obs)
end


function showblock!(io, backend::ShowText, blocks::Tuple, obss::Tuple)
    header = [block isa Pair ? first(block) : "" for block in blocks]
    blocks = [block isa Pair ? last(block) : block for block in blocks]
    data = reshape([PrettyTables.AnsiTextCell(io -> showblock!(IOContext(io, :color => true), backend, block, obs))
                for (block, obs) in zip(blocks, obss)], 1, :)
    pretty_table(io, data; header=header, noheader=all(isempty, header), backend.kwargs...)
end


function showblocks!(io, backend::ShowText, blocks::Tuple, obsss::AbstractVector)
    header = [block isa Pair ? first(block) : "" for block in blocks]
    blocks = [block isa Pair ? last(block) : block for block in blocks]
    rows = []
    for obss in obsss
        row = reshape([AnsiTextCell(
                    io -> showblock!(IOContext(io, :color => true), backend, block, obs))
                for (block, obs) in zip(blocks, obss)], 1, :)
        push!(rows, row)
    end
    tabledata = reduce(vcat, rows)
    pretty_table(io, tabledata; header=header, noheader=all(isempty, header), backend.kwargs...)
end

showblocks!(io, backend::ShowText, block, obss::AbstractVector) =
    showblocks!(io, backend, (block,), map(obs -> (obs,), obss))



# Block implementations



function showblock!(io, ::ShowText, block::Label, obs)
    print(io, obs)
end

function showblock!(io, ::ShowText, block::Continuous, obs)
    print(io, obs)
end

function showblock!(io, ::ShowText, block::LabelMulti, obs)
    print(io, obs)
end

function showblock!(io, ::ShowText, block::OneHotLabel, obs)
    if !(sum(obs) ≈ 1)
        obs = softmax(obs)
    end
    obs = round.(obs; sigdigits=3)
    plot = UnicodePlots.barplot(block.classes, obs, width=20, compact=true)
    print(IOContext(io, :color => true), plot)
end
