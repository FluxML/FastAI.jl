"""
    TextClassification(blocks[, data])

"""
function TextGeneration(blocks::Tuple{<:Paragraph,<:Label}, data; vocab_size = 40000)
    return SupervisedTask(
        blocks,
        (
            Tokenize(),
            setup(EmbedVocabulary, data, vocab_size = vocab_size),
            OneHot()
        )
    )
end

_tasks["textgen"] = (
    id = "textual/textgen",
    name = "Text generation",
    constructor = TextGeneration,
    blocks = (Paragraph),
    category = "unsupervised",
    description = """
        Text generation task.
        """,
    package = @__MODULE__
)