include("../imports.jl")


@testset ExtendedTestSet "DatasetRegistry" begin
    reg = Datasets.DatasetRegistry()
    @testset ExtendedTestSet "registerdataset!" begin
        @test_nowarn Datasets.registerdataset!(
            reg, "mnist_var_size_tiny", () -> datasetpath("mnist_var_size_tiny"))


        # Reregistering should error
        @test_throws ErrorException Datasets.registerdataset!(
            reg, "mnist_var_size_tiny", () -> datasetpath("mnist_var_size_tiny"))
    end

    @testset ExtendedTestSet "listdatasources" begin
        @test length(Datasets.listdatasources(reg)) == 1
    end

    @testset ExtendedTestSet "datasetpath" begin
        @test_nowarn datasetpath(reg, "mnist_var_size_tiny")
    end

    @testset ExtendedTestSet "registerrecipe!" begin
        @test_nowarn Datasets.registerrecipe!(
            reg, "mnist_var_size_tiny", Datasets.ImageClassificationFolders())
    end

    @testset ExtendedTestSet "finddatasets" begin
        @test finddatasets(reg) |> length == 1
        @test finddatasets(reg, name="mnist_var_size_tiny") |> length == 1
        @test finddatasets(reg, blocks=Tuple{Image, Label}) |> length == 1
        @test finddatasets(reg, blocks=Tuple{Image, LabelMulti}) |> length == 0
        @test finddatasets(reg, name="mnist_var_size_tiny", blocks=Tuple{Image, Label}) |> length == 1
        @test finddatasets(reg, name="mnist", blocks=Tuple{Image, Label}) |> length == 0
    end
end
