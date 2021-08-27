include("../imports.jl")

@testset "TabularRegression" begin
    df = DataFrame(A = 1:4, B = ["M", "F", "F", "M"], C = 10:13)
    td = TableDataset(df)
    method = TabularRegression((:B,), (:A,); data = td)
    testencoding(method.encodings, method.blocks)
    DLPipelines.checkmethod_core(method)
    @test_nowarn methodlossfn(method)
    @test_nowarn methodmodel(method, NamedTuple())

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

end