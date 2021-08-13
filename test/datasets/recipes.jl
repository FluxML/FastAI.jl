include("../imports.jl")


function testrecipe(recipe::Datasets.DatasetRecipe, path::AbstractPath)
    data, blocks = loadrecipe(recipe, path)
    return testrecipe(recipe, data, blocks)
end

function testrecipe(recipe::Datasets.DatasetRecipe, data, blocks)
    # `blocks` must be an instance of `recipeblocks(recipe)`
    @test blocks isa Datasets.recipeblocks(recipe)

    # Observations must be compatible with `blocks`
    @test checkblock(blocks, getobs(data, 1))
end


@testset ExtendedTestSet "ImageClassificationFolders" begin
    path = joinpath(datasetpath("mnist_var_size_tiny"), "train")

    @testset ExtendedTestSet "Basic configuration" begin
        recipe = Datasets.ImageClassificationFolders()
        data, blocks = loadrecipe(recipe, path)
        testrecipe(recipe, data, blocks)
        @test blocks[1] isa Image
        @test blocks[2].classes == ["3", "7"]
    end

    @testset ExtendedTestSet "Split configuration" begin
        recipe = Datasets.ImageClassificationFolders(split=true)
        data, blocks = loadrecipe(recipe, path)
        testrecipe(recipe, data["train"], blocks)
    end

    @testset ExtendedTestSet "Error cases" begin
        @testset ExtendedTestSet "Empty directory" begin
            recipe = Datasets.ImageClassificationFolders(split=true)
            @test_throws ErrorException loadrecipe(recipe, mktempdir())
        end

        @testset ExtendedTestSet "Only one label" begin
            recipe = Datasets.ImageClassificationFolders(labelfn=x -> "1")
            @test_throws ErrorException loadrecipe(recipe, path)
        end
    end

end


@testset ExtendedTestSet "ImageSegmentationFolders" begin
    path = datasetpath("camvid_tiny")

    @testset ExtendedTestSet "Basic configuration" begin
        recipe = Datasets.ImageSegmentationFolders()
        data, blocks = loadrecipe(recipe, path)
        testrecipe(recipe, data, blocks)
        @test blocks[1] isa Image
        @test blocks[2] isa Mask
    end

    @testset ExtendedTestSet "Error cases" begin
        @testset ExtendedTestSet "Empty directory" begin
            recipe = Datasets.ImageSegmentationFolders()
            @test_throws ErrorException loadrecipe(recipe, mktempdir())
        end

        @testset ExtendedTestSet "Only one label" begin
            recipe = Datasets.ImageSegmentationFolders(labelfile="idontexist")
            @test_throws ErrorException loadrecipe(recipe, path)
        end
    end
end
