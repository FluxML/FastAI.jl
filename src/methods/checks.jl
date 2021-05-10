"""
    checkmethod_plot(method; model)

"""
function checkmethod_plot(
        method;
        model = mockmodel(method),
        sample = mocksample(method),
        devicefn = cpu,
        context = Training())
    @testset "Plotting interface" begin
        x, y = DLPipelines.encode(method, context, sample)
        ŷ = DLPipelines._predictx(method, model, x, devicefn)

        @test_nowarn plotsample(method, sample; resolution = (200, 200))
        @test_nowarn plotxy(method, x, y; resolution = (200, 200))
        @test_nowarn plotprediction(method, x, ŷ, y; resolution = (200, 200))
    end
end
