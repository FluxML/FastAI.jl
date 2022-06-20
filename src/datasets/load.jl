
function loadfolderdata(
        dir;
        pattern="**",
        splitfn = nothing,
        filterfn = nothing,
        loadfn = nothing)
    data = MLDatasets.FileDataset(identity, dir, pattern)
    if filterfn !== nothing && !isempty(data)
        data = filterobs(filterfn, data)
    end
    if splitfn !== nothing
        data = groupobs(splitfn, data)
    end
    if loadfn !== nothing
        if splitfn === nothing
            data = mapobs(loadfn, data)
        else
            data = Dict(zip(keys(data), map(d -> mapobs(loadfn, d), values(data))))
        end
    end
    return data
end

pathparent(p::String) = splitdir(p)[1]
pathname(p::String) = splitdir(p)[2]
parentname(f) = f |> pathparent |> pathname
grandparentname(f) = f |> pathparent |> pathparent |> pathname
matches(re::Regex) = f -> matches(re, f)
matches(re::Regex, f) = !isnothing(match(re, f))

"""
    loadfile(file)

Load a file from disk into the appropriate format.
"""
loadfile(file::AbstractPath) = loadfile(string(file))
loadfile(file::String) = loadfile(file, Val(Symbol(split("test.txt", '.')[end])))
loadfile(file::String, ::Val) = FileIO.load(file)
