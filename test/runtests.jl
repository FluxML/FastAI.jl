
include("imports.jl")

##

@testset ExtendedTestSet "FastAI.jl" begin
    @testset ExtendedTestSet "datablock.jl" begin
        include("datablock.jl")
    end

    @testset ExtendedTestSet "fasterai.jl" begin
        include("fasterai.jl")
    end

    @testset ExtendedTestSet "encodings/" begin
        @testset ExtendedTestSet "projective.jl" begin
            include("encodings/projective.jl")
        end
        @testset ExtendedTestSet "imagepreprocessing.jl" begin
            include("encodings/imagepreprocessing.jl")
        end
        @testset ExtendedTestSet "keypointpreprocessing.jl" begin
            include("encodings/keypointpreprocessing.jl")
        end
        @testset ExtendedTestSet "many.jl" begin
            include("encodings/many.jl")
        @testset ExtendedTestSet "tabularpreprocessing.jl" begin
            include("encodings/tabularpreprocessing.jl")
        end
    end

    @testset ExtendedTestSet "methods/" begin
        @testset ExtendedTestSet "imageclassification.jl" begin
            include("methods/imageclassification.jl")
        end
        @testset ExtendedTestSet "imagesegmentation.jl" begin
            include("methods/imagesegmentation.jl")
        end
        @testset ExtendedTestSet "imagekeypointregression.jl" begin
            include("methods/imagekeypointregression.jl")
        end
    end

    @testset ExtendedTestSet "datasets/" begin
        @testset ExtendedTestSet "transformations.jl" begin
            include("datasets/transformations.jl")
        end
        @testset ExtendedTestSet "containers.jl" begin
            include("datasets/containers.jl")
        end
        @testset ExtendedTestSet "recipes.jl" begin
            include("datasets/recipes.jl")
        end
        @testset ExtendedTestSet "registry.jl" begin
            include("datasets/registry.jl")
        end
    end

    @testset ExtendedTestSet "training/" begin
        @testset ExtendedTestSet "paramgroups.jl" begin
            include("training/paramgroups.jl")
        end
        @testset ExtendedTestSet "discriminativelrs.jl" begin
            include("training/discriminativelrs.jl")
        end
        @testset ExtendedTestSet "fitonecycle.jl" begin
            include("training/fitonecycle.jl")
        end
        @testset ExtendedTestSet "finetune.jl" begin
            include("training/finetune.jl")
        end
        @testset ExtendedTestSet "lrfind.jl" begin
            include("training/lrfind.jl")
        end
        # TODO: test learning rate finder
    end

    @testset ExtendedTestSet "models/" begin
        @testset ExtendedTestSet "tabularmodel.jl" begin
            include("models/tabularmodel.jl")
        end
    end
end
