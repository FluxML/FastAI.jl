
@testset "Learner" begin
    learn = test_learner()
    rec = Recorder(learn)
    
    run_learner(learn)

    #@show rec["TrainLoss",1,1]
    #@show rec["TrainLoss",16,1]      
    #@show rec["TrainSmoothLoss",1,1]
    #@show rec["TrainSmoothLoss",16,1]
    #@show rec["ValidateLoss",1,1]
    #@show rec["ValidateLoss",16,1]  
    #@show rec["ValidateSmoothLoss",1,1]
    #@show rec["ValidateSmoothLoss",16,1]  

    @test rec["TrainLoss",1,1] > rec["TrainLoss",16,1]      
    @test rec["TrainSmoothLoss",1,1] > rec["TrainSmoothLoss",16,1]
    @test rec["ValidateLoss",1,1] > rec["ValidateLoss",16,1]
    @test rec["ValidateSmoothLoss",1,1] > rec["ValidateSmoothLoss",16,1]      

end
