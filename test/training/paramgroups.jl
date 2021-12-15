
@testset "ParamGroups" begin
    model = Chain(Dense(3, 5), Dense(5, 3))
    paramgroups = FastAI.ParamGroups(FastAI.IndexGrouper([1, 2]), model)

    @test FastAI.getgroup(paramgroups, model[1].weight) == 1
    @test FastAI.getgroup(paramgroups, model[2].weight) == 2
    @test FastAI.getgroup(paramgroups, rand(10)) === nothing
end
