
"""
    savetaskmodel(path, task, model[; force = false])

Save a trained `model` along with a `task` to `path` for later inference.
Use [`loadtaskmodel`](#) for loading both back into a session. If `path`
already exists, only write to it if `force = true`.

If `model` weights are on a GPU, they will be moved to the CPU before saving
so they can be loaded in a non-GPU environment.

[JLD2.jl](https://github.com/JuliaIO/JLD2.jl) is used for serialization.
"""
function savetaskmodel(path, task::LearningTask, model; force = false)
    if !force && isfile(path)
        error("$path already exists. Use `force = true` to overwrite.")
    end
    jldsave(string(path); model=cpu(model), task=task)
end


"""
    loadtaskmodel(path) -> (task, model)

Load a trained `model` along with a `task` from `path` that were saved
using [`savetaskmodel`](#).

[JLD2.jl](https://github.com/JuliaIO/JLD2.jl) is used for serialization.
"""
function loadtaskmodel(path)
    isfile(path) || error("\"$path\" is not an existing file.")
    task, model = jldopen(string(path), "r") do f
        f["task"], f["model"]
    end
    return task, model
end
