function TextClassficationSingle(blocks::Tuple{<:Paragraph, <:Label}, data=nothing)
    return SupervisedTask(
        blocks,
        (
            OneHot()
        )
    )
end

_tasks["textclfsingle"] = (
    id = "textual/textclfsingle",
    name = "Text classification (single-label)",
    constructor = TextClassficationSingle,
    blocks = (Paragraph, Label),
    category = "supervised",
    description = """
        Single-label text classification task where every text has a single
        class label associated with it.
        """,
    package=@__MODULE__,
)

