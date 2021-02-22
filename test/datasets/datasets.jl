include("../imports.jl")

@testset ExtendedTestSet "datasetpath" begin
    @test Datasets.datasetpath("imagenette2-160") isa FilePathsBase.AbstractPath
end

@testset ExtendedTestSet "loaddataset" begin
    @test Datasets.loaddataset("imagenette2-160")
end
