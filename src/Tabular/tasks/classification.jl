
function TabularClassificationSingle(
        blocks::Tuple{<:TableRow, <:Label},
        data)
    tabledata, targetdata = data
    tabledata isa TableDataset || error("`data` needs to be a tuple of a `TableDataset` and targets")

    return SupervisedMethod(
        blocks,
        (
            setup(TabularPreprocessing, blocks[1], tabledata),
            OneHot()
        )
    )
end

"""
    TabularClassificationSingle(blocks, data)

Learning task for single-label tabular classification. Continuous columns are
normalized and missing values are filled, categorical columns are label encoded
taking into account any missing values which might be present. The target value
is predicted from `classes`. `blocks` should be an input and target block
`(TableRow(...), Label(...))`.

    TabularClassificationSingle(classes, tabledata [; catcols, contcols])

Construct learning task with `classes` to classify into and a `TableDataset`
`tabledata`. The column names can be passed in or guessed from the data.
"""
function TabularClassificationSingle(
        classes::AbstractVector,
        tabledata::TableDataset;
        catcols = nothing,
        contcols = nothing)

    blocks = (
        setup(TableRow, tabledata; catcols = catcols, contcols = contcols),
        Label(classes)
    )
    return TabularClassificationSingle(blocks, (tabledata, nothing))
end


# ## Tests

@testset "TabularClassificationSingle [task]" begin
    df = DataFrame(A = 1:4, B = ["M", "F", "F", "M"], C = ["P", "F", "P", "F"])
    td = TableDataset(df)

    task = TabularClassificationSingle(["P", "F"], td; catcols=(:B,), contcols=(:A,), )
    testencoding(getencodings(task), getblocks(task).sample)
    FastAI.checktask_core(task)
    @test_nowarn tasklossfn(task)
    @test_nowarn taskmodel(task)

    @testset "`encodeinput`" begin
        row = mockblock(getblocks(task)[1])

        xtrain = encodeinput(task, Training(), row)
        @test length(xtrain[1]) == length(getblocks(task).input.catcols)
        @test length(xtrain[2]) == length(getblocks(task).input.contcols)

        @test eltype(xtrain[1]) <: Number
    end

    @testset "`encodetarget`" begin
        category = "P"
        y = encodetarget(task, Training(), category)
        @test y ≈ [1, 0]
    end


    @test_nowarn TabularClassificationSingle(["P", "F"], td)
    @test TabularClassificationSingle(["P", "F"], td).blocks[1].catcols == (:B, :C)

    FastAI.test_task_show(task, ShowText(Base.DevNull()))
end
