
@testset "Learner" begin
    learn = test_learner()
    add_cb!(learn,TestCallback())
    
    run_learner(learn)
end
