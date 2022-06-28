"""
    TextClassificationSingle(blocks[, data])

Learning task for single-label text classification. Samples are
preprocessed by applying various textual transforms and classified into one of `classes`.

"""
function TextClassficationSingle(blocks::Tuple{<:Paragraph,<:Label}, data)
    return SupervisedTask(
        blocks,
        (
            Sanitize(),
            Tokenize(),
            setup(EmbedVocabulary, data),
            # EmbedVocabulary(),
            OneHot()
        )
    )
end

_tasks["textclfsingle"] = (
    id="textual/textclfsingle",
    name="Text classification (single-label)",
    constructor=TextClassficationSingle,
    blocks=(Paragraph, Label),
    category="supervised",
    description="""
      Single-label text classification task where every text has a single
      class label associated with it.
      """,
    package=@__MODULE__,
)

# ## Tests

# @testset "TextClassificationSingle [task]" begin
#     task = TextClassificationSingle((Paragraph(), Label{String}(["neg", "pos"])))
#     testencoding(getencodings(task), getblocks(task).sample)
#     FastAI.checktask_core(task)

#     @testset "`encodeinput`" begin
#         paragraph = "A sample paragraph."

#         xtrain = encodeinput(task, Training(), paragraph)
#         @test contains(xtrain, "xxbos")
#         @test lowercase(xtrain) == xtrain
#         @test length(xtrain) == 28
#         @test eltype(xtrain) == Char
#     end

#     @testset "`encodetarget`" begin
#         category = "neg"
#         y = encodetarget(task, Training(), category)
#         @test y ≈ [1, 0]
#     end
# end