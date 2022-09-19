"""
    TextClassification(blocks[, data])

"""
function TextGeneration(blocks::Tuple{<:Paragraph,<:Paragraph}, data; vocab_size = 40000)
    return SupervisedTask(
        blocks,
        (
            Sanitize(),
            Tokenize(),
            setup(EmbedVocabulary, data, vocab_size = vocab_size))
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