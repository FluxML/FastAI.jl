
@testset "Learner" begin
    learn = test_learner()
    add_cb!(learn,TestCallback())
    
    xys = learn |> data_bunch |> train |> one_batch
    lf = loss(learn)
    mf = model(learn)

    function lff(xy)
        x = xy[1]
        p = mf(x)[1]
        y = xy[2]
        return lf(p,y)
    end
    
    init_loss = sum(lff.(xys))
    fit!(learn,16)
    final_loss = sum(lff.(xys))
    @test final_loss < init_loss
end
