function _taskregistry(; name = "Learning tasks")
    registry = Registry((;
                         id = Field(String, name = "ID", formatfn = x -> sprint(show, x)),
                         name = Field(String,
                                      name = "Name",
                                      description = "The name of the learning task",
                                      computefn = (row, key) -> get(row, key, row.id)),
                         blocks = Field(Any,
                                        name = "Block types",
                                        description = "Types of the blocks that are compatible with this task",
                                        filterfn = blocktypesmatch,
                                        formatfn = b -> FeatureRegistries.code_format(_formatblock(b))),
                         category = Field(String,
                                          name = "Category",
                                          description = "Kind of task, e.g. \"supervised\""),
                         description = Field(String,
                                             name = "Description",
                                             optional = true,
                                             description = "More information about the learning task",
                                             formatfn = FeatureRegistries.md_format),
                         constructor = Field(Any,
                                             name = "Learning task",
                                             description = "Function instance to create a corresponding learning task.",
                                             formatfn = FeatureRegistries.code_format),
                         package = Field(Module,
                                         name = "Package",
                                         formatfn = FeatureRegistries.code_format));
                        name,
                        loadfn = row -> row.constructor,
                        description = """
                        A registry for learning tasks. `load`ing an entry will return a function
                        that can be used to construct a `LearningTask` given `blocks`.

                        ```julia
                        taskfn = load(learningtasks(id))
                        task = taskfn(blocks; kwargs...)
                        ```

                        Inspect `?taskfn` for documentation on the arguments that the function accepts.

                        See `datarecipes` to load these datasets in a format compatible with learning
                        tasks.
                        """)
end

const TASKS = _taskregistry()

"""
    learningtasks(; filters...)

Show a registry of available learning tasks. Pass in filters as keyword
arguments to look at a subset.

See also [finding functionality](/documents/docs/discovery.md), [`datasets`](#),
and [`datarecipes`](#). For more information on registries, see
[FeatureRegistries.jl](https://github.com/lorenzoh/FeatureRegistries.jl).

## Examples

Show all available learning tasks:

{cell}
```julia
using FastAI
learningtasks()
```

Show all computer vision tasks:

{cell}
```julia
learningtasks(package=FastVision)
```

Show all classification tasks, i.e. where the target block is a [`Label`](#):

{cell}
```julia
learningtasks(blocks=(Any, Label))
```

Get an explanation of fields in the learning task registry:

{cell}
```julia
info(learningtasks())
```
"""
learningtasks(; kwargs...) = isempty(kwargs) ? TASKS : filter(TASKS; kwargs...)
