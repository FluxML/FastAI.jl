
function TabularRegression(
        blocks::Tuple{<:TableRow, <:Continuous},
        data)
    tabledata, targetdata = data
    tabledata isa TableDataset || error("`data` needs to be a tuple of a `TableDataset` and targets")
    return BlockMethod(
        blocks,
        (setup(TabularPreprocessing, blocks[1], tabledata),),
        outputblock=blocks[2]
    )
end

"""
    TabularRegression(blocks, data)

Learning method for tabular regression. Continuous columns are
normalized and missing values are filled, categorical columns are label encoded
taking into account any missing values which might be present.
 `blocks` should be an input and target block `(TableRow(...), Continuous(...))`.

    TabularRegression(n, tabledata [; catcols, contcols])

Construct learning method with `classes` to classify into and a `TableDataset`
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

@testset "TabularRegression [method]" begin
    df = DataFrame(A = 1:4, B = ["M", "F", "F", "M"], C = 10:13)
    td = TableDataset(df)
    targets = [rand(2) for _ in 1:4]
    method = TabularRegression(2, td; catcols=(:B,), contcols=(:A,))
    testencoding(method.encodings, method.blocks)
    DLPipelines.checkmethod_core(method)
    @test_nowarn methodlossfn(method)
    @test_nowarn methodmodel(method)

    @testset "`encodeinput`" begin
        row = mockblock(method.blocks[1])

        xtrain = encodeinput(method, Training(), row)
        @test length(xtrain[1]) == length(method.blocks[1].catcols)
        @test length(xtrain[2]) == length(method.blocks[1].contcols)

        @test eltype(xtrain[1]) <: Number
    end

    @testset "`encodetarget`" begin
        target = 11
        y = encodetarget(method, Training(), target)
        @test target == y
    end

    @test_nowarn method = TabularRegression(2, td)

    FastAI.test_method_show(method, ShowText(Base.DevNull()))
end
