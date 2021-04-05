using DataFrames
using Tables
using CSV

# FileDataset

struct FileDataset
    tree::FileTree
    nodes::Vector{FileTrees.File}
end

function FileDataset(args...; kwargs...)
    tree = FileTree(args...; kwargs...)
    return FileDataset(tree, FileTrees.files(tree))
end

Base.show(io::IO, data::FileDataset) = print(
    io,
    "FileDataset(\"", data.tree.name, "\", ", nobs(data), " observations)")

LearnBase.nobs(ds::FileDataset) = length(ds.nodes)
LearnBase.getobs(ds::FileDataset, idx::Int) = path(ds.nodes[idx])


# File utilities

"""
    loadfile(file)

Load a file from disk into the appropriate format.
"""
function loadfile(file::String)
    if isimagefile(file)
        # faster image loading
        return FileIO.load(file, view = true)
    else
        return FileIO.load(file)
    end
end

loadfile(file::AbstractPath) = loadfile(string(file))
loadfile(file::FileTrees.File) = loadfile(path(file))


isimagefile(file::AbstractPath) = isimagefile(string(file))
isimagefile(file::File) = isimagefile(file.name)
isimagefile(file::String) = occursin(IMAGEFILE_REGEX, lowercase(file))
const IMAGEFILE_REGEX = r"\.(gif|jpe?g|tiff?|png|webp|bmp)$"


struct TableDataset{T}
    table::T #Should implement Tables.jl interface
    TableDataset{T}(table::T) where T = Tables.istable(table) ? new{T}(table) : error("Object doesn't implement Tables.jl interface")
end

TableDataset(path::String) = TableDataset(DataFrame(CSV.File(path)))
TableDataset(table::T) where {T} = TableDataset{T}(table)

function LearnBase.getobs(dataset::FastAI.Datasets.TableDataset{T}, idx) where {T}
    if Tables.rowaccess(dataset.table)
        for (index, row) in enumerate(Tables.rows(dataset.table))
            if index==idx
                return row
            end
        end
    elseif Tables.columnaccess(dataset.table)
        rowvals = []
        for i in 1:(length ∘ Tables.columnnames)(dataset.table)
            append!(rowvals, Tables.getcolumn(dataset.table, i)[idx])
        end
        return rowvals
    else error("The Tables.jl implementation used should have either rowaccess or columnaccess.")
    end
end

function LearnBase.nobs(dataset::FastAI.Datasets.TableDataset{T}) where {T}
    if Tables.rowaccess(dataset.table)
        return length(Tables.rows(dataset.table))
    elseif Tables.columnaccess(dataset.table)
        return length(Tables.getcolumn(dataset.table, 1))
    else error("The Tables.jl implementation used should have either rowaccess or columnaccess.")
    end
end