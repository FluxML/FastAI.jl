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
LearnBase.getobs(ds::FileDataset, idx::Int) = Path(path(ds.nodes[idx]))


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

#TableDataset

struct TableDataset{T}
    table::T #Should implement Tables.jl interface
    TableDataset{T}(table::T) where T = Tables.istable(table) ? new{T}(table) : error("Object doesn't implement Tables.jl interface")
end

TableDataset(table::T) where {T} = TableDataset{T}(table)
TableDataset(path::AbstractPath) = TableDataset(DataFrame(CSV.File(path)))

function LearnBase.getobs(dataset::FastAI.Datasets.TableDataset, idx)
    if Tables.rowaccess(dataset.table)
        row, _ = Iterators.peel(Iterators.drop(Tables.rows(dataset.table), idx - 1))
        return row
    elseif Tables.columnaccess(dataset.table)
        colnames = Tables.columnnames(dataset.table)
        rowvals = [Tables.getcolumn(dataset.table, i)[idx] for i in 1:length(colnames)]
        return (; zip(colnames, rowvals)...)
    else 
        error("The Tables.jl implementation used should have either rowaccess or columnaccess.")
    end
end

function LearnBase.nobs(dataset::TableDataset)
    if Tables.columnaccess(dataset.table)
        return length(Tables.getcolumn(dataset.table, 1))
    elseif Tables.rowaccess(dataset.table)
        return length(Tables.rows(dataset.table)) # length might not be defined, but has to be for this to work.
    else 
        error("The Tables.jl implementation used should have either rowaccess or columnaccess.")
    end
end

LearnBase.getobs(dataset::TableDataset{<:DataFrame}, idx) = dataset.table[idx, :]
LearnBase.nobs(dataset::TableDataset{<:DataFrame}) = nrow(dataset.table)

LearnBase.getobs(dataset::TableDataset{<:CSV.File}, idx) = dataset.table[idx]
LearnBase.nobs(dataset::TableDataset{<:CSV.File}) = length(dataset.table)
