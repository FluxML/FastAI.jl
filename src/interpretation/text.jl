

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
    stdout,
    (; hlines=hlines, alignment=alignment, kwargs...))

createhandle(backend::ShowText) = backend.io


function showblock!(io, backend::ShowText, (title, block)::Pair, data)
    printstyled(io, title, bold=true)
    println(io)
    showblock!(io, backend, block, data)
end


function showblock!(io, backend::ShowText, blocks::Tuple, datas::Tuple)
    header = [block isa Pair ? first(block) : "" for block in blocks]
    blocks = [block isa Pair ? last(block) : block for block in blocks]
    data = reshape([AnsiTextCell(io -> showblock!(IOContext(io, :color => true), backend, block, data))
                for (block, data) in zip(blocks, datas)], 1, :)
    pretty_table(io, data; header=header, noheader=all(isempty, header), backend.kwargs...)
end


function showblocks!(io, backend::ShowText, blocks::Tuple, datass::AbstractVector)
    header = [block isa Pair ? first(block) : "" for block in blocks]
    blocks = [block isa Pair ? last(block) : block for block in blocks]
    rows = []
    for datas in datass
        row = reshape([AnsiTextCell(io -> showblock!(io, backend, block, data))
                for (block, data) in zip(blocks, datas)], 1, :)
        push!(rows, row)
    end
    tabledata = reduce(vcat, rows)
    pretty_table(io, tabledata; header=header, noheader=all(isempty, header), backend.kwargs...)
end



# Block implementations


function showblock!(io, ::ShowText, block::Image{2}, data)
    ImageInTerminal.imshow(io, data)
end

function showblock!(io, ::ShowText, block::Label, data)
    print(io, data)
end

function showblock!(io, ::ShowText, block::Continuous, data)
    print(io, data)
end

function showblock!(io, ::ShowText, block::LabelMulti, data)
    print(io, data)
end

function showblock!(io, ::ShowText, block::OneHotTensor{0}, data)
    plot = UnicodePlots.barplot(block.classes, data)
    print(IOContext(io, :color => true), plot)
end

function showblock!(io, ::ShowText, block::TableRow, data)
    rowdata = vcat(
        [data[col] for col in block.catcols],
        [data[col] for col in block.contcols],
    )
    rownames = [block.catcols..., block.contcols...]
    tabledata = hcat(rownames, rowdata)
    pretty_table(
        io, tabledata;
        alignment=[:r, :l],
        highlighters=Highlighter((data, i, j) -> (j == 2), bold=true),
        noheader=true, tf=PrettyTables.tf_borderless,)
end
