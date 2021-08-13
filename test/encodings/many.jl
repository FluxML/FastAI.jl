include("../imports.jl")


@testset ExtendedTestSet "`Many`" begin
    enc = ImagePreprocessing()
    block = Many(Image{2}())
    FastAI.testencoding(enc, block)
end
