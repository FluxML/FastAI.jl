
# ## `DatasetLoader` interface

"""
    abstract type DatasetLoader

A `DatasetLoader` defines how a dataset can made available and loaded.
See [`DataDepLoader`](#) as an example.

A `DatasetLoader` has to implement the following functions:

- [`loaddata`](#)
- [`makeavailable`](#)
- [`isavailable`](#)
"""
abstract type DatasetLoader end

function makeavailable end
function isavailable end
function loaddata end


# ## Loader for `DataDep`s


"""
    struct DataDepLoader(datadep) <: DatasetLoader

A dataset loader that uses DataDeps.jl to load datasets.
The DataDep has to be registered before creating the loader, and will
error otherwise.
"""
struct DataDepLoader <: DatasetLoader
    datadep::String
    function DataDepLoader(datadep)
        if !haskey(DataDeps.registry, datadep)
            throw(ArgumentError("DataDep \"$datadep\" does not exist."))
        end
        return new(datadep)
    end
end


isavailable(loader::DataDepLoader) = !isnothing(DataDeps.try_determine_load_path(loader.datadep, @__FILE__))

function makeavailable(loader::DataDepLoader)
    return DataDeps.resolve(loader.datadep, @__FILE__)
end

function loaddata(loader::DataDepLoader)
    isavailable(loader) || makeavailable(loader)
    return DataDeps.resolve(loader.datadep, @__FILE__)
end
