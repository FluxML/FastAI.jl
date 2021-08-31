include("imports.jl")

@testset ExtendedTestSet "`methodlearner`" begin
    method = BlockMethod((Label(1:2), Label(1:2)), (OneHot(),))
    data = (rand(1:2, 1000), rand(1:2, 1000))
    @test_nowarn learner = methodlearner(method, data, model=identity)

    @testset ExtendedTestSet "batch sizes" begin
        learner = methodlearner(method, data, model=identity, batchsize=100)
        @test length(learner.data.training) == 8
        @test length(learner.data.validation) == 1

        learner = methodlearner(method, data, model=identity, pctgval=0.4, batchsize=100)
        @test length(learner.data.training) == 6
        @test length(learner.data.validation) == 2

        learner = methodlearner(method, data, model=identity, batchsize=100, validbsfactor=1)
        @test length(learner.data.training) == 8
        @test length(learner.data.validation) == 2
    end

    @testset ExtendedTestSet "callbacks" begin
        learner = methodlearner(
            method, data, model=identity,
            callbacks=[ToGPU(), Checkpointer(mktempdir())])
        @test !isnothing(FluxTraining.getcallback(learner, Checkpointer))

    end
end


@testset ExtendedTestSet "`blockbackbone`" begin
    @test_nowarn FastAI.blockbackbone(FastAI.ImageTensor{2}(3))

    @test_nowarn FastAI.blockbackbone(FastAI.EncodedTableRow((:x,), (:y,), Dict(:x => [1, 2])))
end


@testset ExtendedTestSet "`blockmodel`" begin
    method = ImageClassificationSingle((Image{2}(), Label(1:2)))
    @test_nowarn methodmodel(method)
end
