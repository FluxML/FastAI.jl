
include("imports.jl")

##

@testset ExtendedTestSet "FastAI.jl" begin
    include("steps/projective.jl")
    include("steps/imagepreprocessing.jl")
    include("methods/imageclassification.jl")
    include("methods/imagesegmentation.jl")
    include("datasets/transformations.jl")
    include("datasets/containers.jl")
    include("training.jl")
end
