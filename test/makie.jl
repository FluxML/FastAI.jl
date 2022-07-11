
@testset "ShowMakie" begin
    @testset "ImageClassificationSingle" begin
        task = ImageClassificationSingle((16, 16), [1, 2])
        FastAI.test_task_show(task, ShowMakie())
    end
    @testset "ImageClassificationMulti" begin
        task = ImageClassificationMulti((16, 16), [1, 2])
        FastAI.test_task_show(task, ShowMakie())
    end
    @testset "ImageSegmentation" begin
        task = ImageSegmentation((16, 16), [1, 2])
        FastAI.test_task_show(task, ShowMakie())
    end
    @testset "ImageKeypointRegression" begin
        task = ImageKeypointRegression((16, 16), 10)
        FastAI.test_task_show(task, ShowMakie())
    end
end
