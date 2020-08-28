
@testset "Recorder" begin
    learner = test_learner()

    tlr = Recorder.TrainLoss()
    vlr = Recorder.ValidateLoss()
    tslr = Recorder.SmoothTrainLoss()
    vslr = Recorder.SmoothValidateLoss()

    add_cb!(learner,tlr)
    add_cb!(learner,vlr)
    add_cb!(learner,tslr)
    add_cb!(learner,vslr)

    run_learner(learner)

    @test tlr[1,1] > tlr[16,1]      
    @test vlr[1,1] > vlr[16,1]
    @test tslr[1,1] > tslr[16,1]
    @test vslr[1,1] > vslr[16,1]
end
