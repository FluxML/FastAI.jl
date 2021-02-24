include("../imports.jl")

@testset ExtendedTestSet "datasetpath" begin
    @test Datasets.datasetpath("mnist_var_size_tiny") isa FilePathsBase.AbstractPath
end

@testset ExtendedTestSet "loaddataset" begin
    @test_nowarn Datasets.loadtaskdata(
        Datasets.datasetpath("mnist_var_size_tiny"),
        ImageClassificationTask)
end
