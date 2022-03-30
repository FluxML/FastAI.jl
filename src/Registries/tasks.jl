function _taskregistry(; name = "Learning tasks")
    registry = Registry(
        name,
        (;
            id = Field(
                String,
                "ID"),
            name = Field(
                String,
                "Name",
                description = "The name of the learning task",
                computefn = (row, key) -> get(row, key, row.id)),
            blocks = Field(
                Any,
                "Block types",
                description = "Types of the blocks that are compatible with this task",
                getfilterfn = filterblocks,
                formatfn = _formatblock),
            category = Field(
                String,
                "Category",
                description = "Kind of task, e.g. \"supervised\""),
            description = Field(
                String,
                "Description",
                optional = true,
                description = "More information about the learning task",
                formatfn=x -> ismissing(x) ? x : Markdown.parse(x)),
            constructor = Field(
                Any,
                "Learning task",
                description = "Function instance to create a corresponding learning task."),
            package = Field(
                Module,
                "Package"),
        );
        loadfn = row -> row.constructor,
    )
end

TASKS = _taskregistry()
