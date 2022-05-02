function _taskregistry(; name = "Learning tasks")
    registry = Registry(
        (;
            id = Field(String, name = "ID"),
            name = Field(
                String,
                name = "Name",
                description = "The name of the learning task",
                computefn = (row, key) -> get(row, key, row.id)),
            blocks = Field(
                Any,
                name = "Block types",
                description = "Types of the blocks that are compatible with this task",
                filterfn = blocktypesmatch,
                formatfn = _formatblock),
            category = Field(
                String,
                name = "Category",
                description = "Kind of task, e.g. \"supervised\""),
            description = Field(
                String,
                name = "Description",
                optional = true,
                description = "More information about the learning task",
                formatfn=x -> ismissing(x) ? x : Markdown.parse(x)),
            constructor = Field(
                Any,
                name = "Learning task",
                description = "Function instance to create a corresponding learning task."),
            package = Field(
                Module,
                name = "Package"),
        );
        name,
        loadfn = row -> row.constructor,
    )
end

const TASKS = _taskregistry()
learningtasks(; kwargs...) = isempty(kwargs) ? TASKS : find(TASKS; kwargs...)
