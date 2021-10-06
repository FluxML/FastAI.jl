include("../imports.jl")


@testset ExtendedTestSet "ImageKeypointRegression" begin
    method = ImageKeypointRegression((16, 16), 10)
    DLPipelines.checkmethod_core(method)
    @testset "Show backends" begin
        @testset "ShowText" begin
            FastAI.test_method_show(method, ShowText(Base.DevNull()))
        end
        @testset "ShowMakie" begin
            FastAI.test_method_show(method, ShowMakie())
        end
    end
end
