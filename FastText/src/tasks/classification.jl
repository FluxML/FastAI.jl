"""
    TextClassificationSingle(blocks[, data])

Learning task for single-label text classification. Samples are
preprocessed by applying various textual transforms and classified into one of `classes`.

"""
function TextClassificationSingle(blocks::Tuple{<:Paragraph,<:Label}, data; vocab_size = 40000)
    blocks = (blocks[1], Named(:target, blocks[2]))
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

_tasks["textclfsingle"] = (
    id = "textual/textclfsingle",
    name = "Text classification (single-label)",
    constructor = TextClassificationSingle,
    blocks = (Paragraph, Label),
    category = "supervised",
    description = """
        Single-label text classification task where every text has a single
        class label associated with it.
        """,
    package = @__MODULE__
)

# ## Tests

@testset "TextClassificationSingle [task]" begin
    task = TextClassificationSingle((Paragraph(), Label{String}(["neg", "pos"])), [("A good review", "pos")])
    testencoding(getencodings(task), getblocks(task).sample, ("A good review", "pos"))
    FastAI.checktask_core(task, sample = ("A good review", "pos"))

    @testset "`encodeinput`" begin
        paragraph = "A good review"

        xtrain = encodeinput(task, Training(), paragraph)
        @test eltype(xtrain) == Int64
    end

    @testset "`encodetarget`" begin
        category = "pos"
        y = encodetarget(task, Training(), category)
        @test y ≈ [0, 1]
    end
end
