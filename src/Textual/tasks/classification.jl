"""
    TextClassificationSingle(blocks[, data])

Learning task for single-label text classification. Samples are
preprocessed by applying various textual transforms and classified into one of `classes`.

"""
function TextClassficationSingle(blocks::Tuple{<:Paragraph, <:Label}, data=nothing)
    return SupervisedTask(
        blocks,
        (
            TextEncoding(),
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

