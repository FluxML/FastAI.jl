include("../imports.jl")

@testset "TabularClassificationSingle" begin
    df = DataFrame(A = 1:4, B = ["M", "F", "F", "M"], C = ["P", "F", "P", "F"])
    td = TableDataset(df)
    method = TabularClassificationSingle((:B,), (:A,), ["P", "F"]; data = td)
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
        category = "P"
        y = encodetarget(method, Training(), category)
        @test y ≈ [1, 0]
    end

end
