
function TabularRegression(
        blocks::Tuple{<:TableRow, <:Continuous},
        data)
    tabledata, targetdata = data
    tabledata isa TableDataset || error("`data` needs to be a tuple of a `TableDataset` and targets")
    return SupervisedTask(
        blocks,
        (setup(TabularPreprocessing, blocks[1], tabledata),),
        ŷblock=blocks[2],
    )
end

"""
    TabularRegression(blocks, data)

Learning task for tabular regression. Continuous columns are
normalized and missing values are filled, categorical columns are label encoded
taking into account any missing values which might be present.
 `blocks` should be an input and target block `(TableRow(...), Continuous(...))`.

    TabularRegression(n, tabledata [; catcols, contcols])

Construct learning task with `classes` to classify into and a `TableDataset`
`tabledata`. The column names can be passed in or guessed from the data. The
regression target is a vector of `n` values.
"""
function TabularRegression(
        n::Int,
        tabledata::TableDataset;
        catcols = nothing,
        contcols = nothing)
    blocks = (
        setup(TableRow, tabledata; catcols=catcols, contcols=contcols),
        Continuous(n)
    )
    return TabularRegression(blocks, (tabledata, nothing))
end


# ## Tests

@testset "TabularRegression [task]" begin
    df = DataFrame(A = 1:4, B = ["M", "F", "F", "M"], C = 10:13)
    td = TableDataset(df)
    targets = [rand(2) for _ in 1:4]
    task = TabularRegression(2, td; catcols=(:B,), contcols=(:A,))
    testencoding(getencodings(task), getblocks(task).sample)
    FastAI.checktask_core(task)
    @test_nowarn tasklossfn(task)
    @test_nowarn taskmodel(task)

    @testset "`encodeinput`" begin
        row = mockblock(getblocks(task).input)

        xtrain = encodeinput(task, Training(), row)
        @test length(xtrain[1]) == length(getblocks(task).input.catcols)
        @test length(xtrain[2]) == length(getblocks(task).input.contcols)

        @test eltype(xtrain[1]) <: Number
    end

    @testset "`encodetarget`" begin
        target = 11
        y = encodetarget(task, Training(), target)
        @test target == y
    end

    @test_nowarn task = TabularRegression(2, td)

    FastAI.test_task_show(task, ShowText(Base.DevNull()))
end
