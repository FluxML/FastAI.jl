
include("imports.jl")

##

@testset ExtendedTestSet "FastAI.jl" begin
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
    end

    @testset ExtendedTestSet "methods/" begin
        @testset ExtendedTestSet "imageclassification.jl" begin
            include("methods/imageclassification.jl")
        end
        @testset ExtendedTestSet "imagesegmentation.jl" begin
            include("methods/imagesegmentation.jl")
        end
        @testset ExtendedTestSet "singlekeypointregression.jl" begin
            include("methods/singlekeypointregression.jl")
        end
    end

    @testset ExtendedTestSet "datasets/" begin
        @testset ExtendedTestSet "transformations.jl" begin
            include("datasets/transformations.jl")
        end
        @testset ExtendedTestSet "containers.jl" begin
            include("datasets/containers.jl")
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
