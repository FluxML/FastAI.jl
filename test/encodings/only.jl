include("../imports.jl")

@testset ExtendedTestSet "`Only`" begin
    tfm = ImagePreprocessing()
    only = Only(:name, tfm)
    block = Named(:name, Image{2}())
    data = mockblock(block)
    testencoding(only, block, data)
    testencoding(only, (block, Image{2}()), (data, data))

    @test encodedblock(only, Image{2}()) isa Nothing
    @test !isnothing(encodedblock(only, Named(:name, Image{2}())))
end
