
@testset "Learner" begin
    learner = test_learner()
    tlr = TrainLossRecorder()
    vlr = ValidateLossRecorder()
    tslr = TrainSmoothLossRecorder()
    vslr = ValidateSmoothLossRecorder()
    learner.add_cb!(learner,tlr)
    learner.add_cb!(learner,vlr)
    learner.add_cb!(learner,tslr)
    learner.add_cb!(learner,vslr)
    run_learner(learner)

    @test tlr[1,1] > tlr[16,1]      
    @test vlr[1,1] > vlr[16,1]
    @test tslr[1,1] > tslr[16,1]
    @test vslr[1,1] > vslr[16,1]
end
