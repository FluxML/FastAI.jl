
@testset "Composition" begin
    encodings = (ProjectiveTransforms((5, 5)), ImagePreprocessing(), OneHot())
    blocks = (Image{2}(), Label(1:10))
    data = (rand(RGB{N0f8}, 10, 10), 7)
    testencoding(encodings, blocks, data)
end
