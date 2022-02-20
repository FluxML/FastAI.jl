

@testset "ShowMakie" begin
    @testset "ImageClassificationSingle" begin
        method = ImageClassificationSingle((16, 16), [1, 2])
        FastAI.test_method_show(method, ShowMakie())
    end
    @testset "ImageClassificationMulti" begin
        method = ImageClassificationMulti((16, 16), [1, 2])
        FastAI.test_method_show(method, ShowMakie())
    end
    @testset "ImageSegmentation" begin
        method = ImageSegmentation((16, 16), [1, 2])
        FastAI.test_method_show(method, ShowMakie())
    end
    @testset "ImageKeypointRegression" begin
        method = ImageKeypointRegression((16, 16), 10)
        FastAI.test_method_show(method, ShowMakie())
    end
end
