"""
    checkmethod_plot(method)

"""
function checkmethod_plot(
        method::LearningMethod;
        model = mockmodel(method),
        sample = mocksample(method),
        devicefn = cpu,
        context = Training())
    @testset "Plotting interface" begin
        x, y = DLPipelines.encode(method, context, sample)
        ŷ = DLPipelines._predictx(method, model, x, devicefn)

        @testset "plotsample!" begin
            @test_nowarn plotsample(method, sample; resolution = (200, 200))
        end
        @testset "plotxy!" begin
            @test_nowarn plotxy(method, x, y; resolution = (200, 200))
        end
        @testset "plotprediction!" begin
            @test_nowarn plotprediction(method, x, ŷ, y; resolution = (200, 200))
        end
    end
end
