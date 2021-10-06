include("../imports.jl")

@testset "TabularRegression" begin
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
