"""
    TextClassification(blocks[, data])

"""
function TextGeneration(blocks::Tuple{<:Paragraph,<:Paragraph}, data; vocab_size = 40000)
    blocks = blocks[1], Named(:target, blocks[2])
    return SupervisedTask(
        blocks,
        (
            Sanitize(),
            Tokenize(),
            setup(EmbedVocabulary, data, vocab_size = vocab_size),
            Only(:target, OneHot())
        )
    )
end

_tasks["textgen"] = (
    id = "textual/textgen",
    name = "Text generation",
    constructor = TextGeneration,
    blocks = (Paragraph, Paragraph),
    category = "unsupervised",
    description = """
        Text generation task.
        """,
    package = @__MODULE__
)