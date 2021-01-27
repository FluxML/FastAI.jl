

# FileDataset

struct FileDataset
    tree::FileTree
    nodes::Vector{FileTrees.File}
end

function FileDataset(args...; kwargs...)
    tree = FileTree(args...; kwargs...)
    return FileDataset(tree, FileTrees.files(tree))
end

LearnBase.nobs(ds::FileDataset) = length(ds.nodes)
LearnBase.getobs(ds::FileDataset, idx::Int) = ds.nodes[idx]


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


isimagefile(file::File) = isimagefile(file.name)
isimagefile(file::String) = occursin(IMAGEFILE_REGEX, lowercase(file))
const IMAGEFILE_REGEX = r"\.(gif|jpe?g|tiff?|png|webp|bmp)$"
