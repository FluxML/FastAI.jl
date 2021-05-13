
"""
    savemethodmodel(path, method, model[; force = false])

Save a trained `model` along with a `method` to `path` for later inference.
Use [`loadmethodmodel`](#) for loading both back into a session. If `path`
already exists, only write to it if `force = true`.

If `model` weights are on a GPU, they will be moved to the CPU before saving
so they can be loaded in a non-GPU environment.
"""
function savemethodmodel(path, method::LearningMethod, model; force = false)
    if !force && isfile(path)
        error("$path already exists. Use `force = true` to overwrite.")
    end
    jldsave(string(path); model=cpu(model), method=method)
end


"""
    loadmethodmodel(path) -> (method, model)

Load a trained `model` along with a `method` from `path` that were saved
using [`savemethodmodel`](#).
"""
function loadmethodmodel(path)
    isfile(path) || error("\"$path\" is not an existing file.")
    method, model = jldopen(string(path), "r") do f
        f["method"], f["model"]
    end
    return method, model
end
