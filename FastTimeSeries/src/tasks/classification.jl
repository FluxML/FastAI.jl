"""
    TSClassificationSingle(blocks[, data])

Learning task for single-label time-series classification. Samples are standardized 
and classified into one of `classes`.
"""

function TSClassificationSingle(blocks::Tuple{<:TimeSeriesRow,<:Label}, data)
    return SupervisedTask(
        blocks,
        (
            OneHot(),
            setup(TSPreprocessing, blocks[1], data[1].table)
        )
    )    
end

_tasks["tsclfsingle"] = (
    id = "timeseries/single",
    name = "Time-Series Classification (single-label)",
    constructor = TSClassificationSingle,
    blocks = (TimeSeriesRow, Label),
    category = "supervised",
    description = """
        Time-Series classification task where every time-series has a single 
        class label associated with it.
        """,
    package = @__MODULE__,
)

# ## Tests

@testset "TabularClassificationSingle [task]" begin
    data, blocks = load(datarecipes()["atrial"])

    task = TSClassificationSingle(blocks, data)
    FastAI.testencoding(task.encodings, task.blocks.sample)
    FastAI.checktask_core(task)
    @test_nowarn tasklossfn(task)
    
    @testset "`encodeinput`" begin 
        row = FastAI.mockblock(task.blocks[1])

        xtrain = FastAI.encodeinput(task, Training(), row);
        @test length(xtrain[1,:]) == task.blocks.input.obslength
        @test length(xtrain[:,1]) == task.blocks.input.nfeatures

        @test eltype(xtrain[1,:]) <: Number
    end

    @testset "`encodetarget`" begin
        category = "s"
        y = FastAI.encodetarget(task, Training(), category)
        @test y ≈ [0, 1, 0]
    end

    FastAI.test_task_show(task, ShowText(Base.DevNull()))
end