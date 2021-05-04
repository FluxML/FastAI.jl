include("../imports.jl")

@testset ExtendedTestSet "`DiscriminativeLRs`" begin
    model = Chain(Dense(3, 5), Dense(5, 3))
    pg = ParamGroups(IndexGrouper([1, 2]), model)
    o = Optimiser(
        DiscriminativeLRs(pg, Dict(1 => 0., 2 => 1.)),
        Descent(0.1)
    )
    x1 = model[1].W
    x2 = model[2].W
    # Weight of layer 1 has zeroed gradient
    @test apply!(o, x1, ones(size(x1))) == zeros(size(x1))
    # Weight of layer 2 has regular gradient
    @test apply!(o, x2, ones(size(x2))) != fill(0.1, size(x1))
end
