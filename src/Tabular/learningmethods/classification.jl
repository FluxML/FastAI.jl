
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

Learning method for single-label tabular classification. Continuous columns are
normalized and missing values are filled, categorical columns are label encoded
taking into account any missing values which might be present. The target value
is predicted from `classes`. `blocks` should be an input and target block
`(TableRow(...), Label(...))`.

    TabularClassificationSingle(classes, tabledata [; catcols, contcols])

Construct learning method with `classes` to classify into and a `TableDataset`
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

@testset "TabularClassificationSingle [method]" begin
    df = DataFrame(A = 1:4, B = ["M", "F", "F", "M"], C = ["P", "F", "P", "F"])
    td = TableDataset(df)

    method = TabularClassificationSingle(["P", "F"], td; catcols=(:B,), contcols=(:A,), )
    testencoding(getencodings(method), getblocks(method).sample)
    DLPipelines.checkmethod_core(method)
    @test_nowarn methodlossfn(method)
    @test_nowarn methodmodel(method)

    @testset "`encodeinput`" begin
        row = mockblock(getblocks(method)[1])

        xtrain = encodeinput(method, Training(), row)
        @test length(xtrain[1]) == length(getblocks(method).input.catcols)
        @test length(xtrain[2]) == length(getblocks(method).input.contcols)

        @test eltype(xtrain[1]) <: Number
    end

    @testset "`encodetarget`" begin
        category = "P"
        y = encodetarget(method, Training(), category)
        @test y ≈ [1, 0]
    end


    @test_nowarn TabularClassificationSingle(["P", "F"], td)
    @test TabularClassificationSingle(["P", "F"], td).blocks[1].catcols == (:B, :C)

    FastAI.test_method_show(method, ShowText(Base.DevNull()))
end
