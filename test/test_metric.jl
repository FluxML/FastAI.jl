using FastAI
using Test

@testset "Metric" begin

mutable struct TstLearner
    pb
    yb
end

TstLearner() = TstLearner([],[])
current_batch(l:: TstLearner) = (ps,ys)

_l2_mean(x,y) = float.(x)-float.(y) |> pow(2) |> mean

function compute_val(met, x1, x2)
    reset(met)
    vals = [0,6,15,20]
    learn = TstLearner()
    for i in range(3) 
        learn.ps,learn.ys = x1[vals[i]:vals[i+1]],x2[vals[i]:vals[i+1]]
        met.accumulate(met,learn)
    end
    return value(met)
end



@testset "AvgMetric" begin

    lrn = TstLearner()
    met = AvgMetric(x,y -> mean(abs(x-y)))
    met.reset()
    ps,ys = randn(100),randn(100)
    for i in 1:100 
        lrn.pb,lrn.yb = ps[i:i+25],ys[i:i+25]
        accumulate(met,lrn)
    end
    @test value(met) ≈ mean(abs(ps-ys)) 

end

@testset "AvgLoss" begin
#=
tst = AvgLoss()
t = torch.randn(100)
tst.reset()
for i in range(0,100,25): 
    learn.yb,learn.loss = t[i:i+25],t[i:i+25].mean()
    tst.accumulate(learn)
test_close(tst.value, t.mean())
=#
end

"""
@testset "AccumMetric" begin
    #Go through a fake cycle with various batch sizes and computes the value of met
    x1,x2 = randn(20,5),torch.randn(20,5)
    tst = AccumMetric(_l2_mean)
    @test compute_val(tst, x1, x2) ≈ _l2_mean(x1, x2)
    @test tst.preds == x1.view(-1)
    @test tst.targs == x2.view(-1)

        #test argmax
        x1,x2 = torch.randn(20,5),torch.randint(0, 5, (20,))
        tst = AccumMetric(_l2_mean, dim_argmax=-1)
        test_close(compute_val(tst, x1, x2), _l2_mean(x1.argmax(dim=-1), x2))

        #test thresh
        x1,x2 = torch.randn(20,5),torch.randint(0, 2, (20,5)).bool()
        tst = AccumMetric(_l2_mean, thresh=0.5)
        test_close(compute_val(tst, x1, x2), _l2_mean((x1 >= 0.5), x2))

        #test sigmoid
        x1,x2 = torch.randn(20,5),torch.randn(20,5)
        tst = AccumMetric(_l2_mean, activation=ActivationType.Sigmoid)
        test_close(compute_val(tst, x1, x2), _l2_mean(torch.sigmoid(x1), x2))

        #test to_np
        x1,x2 = torch.randn(20,5),torch.randn(20,5)
        tst = AccumMetric(lambda x,y: isinstance(x, np.ndarray) and isinstance(y, np.ndarray), to_np=True)
        assert compute_val(tst, x1, x2)

        #test invert_arg
        x1,x2 = torch.randn(20,5),torch.randn(20,5)
        tst = AccumMetric(lambda x,y: torch.sqrt(x.pow(2).mean()))
        test_close(compute_val(tst, x1, x2), torch.sqrt(x1.pow(2).mean()))
        tst = AccumMetric(lambda x,y: torch.sqrt(x.pow(2).mean()), invert_arg=True)
        test_close(compute_val(tst, x1, x2), torch.sqrt(x2.pow(2).mean()))
    
    """
end