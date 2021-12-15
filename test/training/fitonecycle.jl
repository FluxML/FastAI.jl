@testset "fitonecycle!" begin
    learner = testlearner(Recorder())
    @test_nowarn fitonecycle!(learner, 5)
end


import FastAI: decay_optim
@testset "decay_optim" begin
    optim = ADAM()
    @test decay_optim(optim, 0.1) isa Optimiser
    @test decay_optim(Optimiser(ADAM(), ADAM()), 0.1) isa Optimiser
    @test decay_optim(optim, 0.1).os[1] isa WeightDecay
    o = decay_optim(Optimiser(ADAM(), WeightDecay(.5)), .1)
    @test o.os[1] isa WeightDecay
    @test o.os[2] isa ADAM
end
