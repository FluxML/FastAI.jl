using FastAI
using Statistics
using Test
using Infiltrator

@testset "Metric" begin

mutable struct TestLearner <: AbstractLearner
    pb
    yb
    loss
end

TestLearner() = TestLearner([],[],0.0)
FastAI.batch_size(l::TestLearner) = length(l.yb)
FastAI.pb(l:: TestLearner) = l.pb
FastAI.yb(l:: TestLearner) = l.yb
FastAI.loss(l::TestLearner) = l.loss

using FastAI: AvgMetric, reset, accumulate, value, name

_l2_mean(x,y) = float.(x)-float.(y) |> pow(2) |> mean

function compute_val(met, x1, x2)
    reset(met)
    vals = [0,6,15,20]
    lrn = TestLearner()
    for i in range(3) 
        lrn.ps,lrn.ys = x1[vals[i]:vals[i+1]],x2[vals[i]:vals[i+1]]
        accumulate(met,lrn)
    end
    return value(met)
end

function is_close(a,b,eps=1e-5)
    return (a-b)^2 < eps^2
end

@testset "AvgMetric" begin
    lrn = TestLearner()
    met = AvgMetric((x,y) -> mean(abs.(x-y)))
    reset(met)
    ps = randn(100)
    ys = randn(100)
    n = 25
    for i in 1:n:100
        lrn.pb = ps[i:i+n-1]
        lrn.yb = ys[i:i+n-1]
        accumulate(met,lrn)
    end
    @test is_close(value(met), mean(abs.(ps-ys))) 
end

@testset "AvgSmoothLoss" begin
    lrn = TestLearner()
    met = AvgSmoothLoss(0.98)
    reset(met)
    t = randn(100)
    val = 0.0
    n = 25
    for i in 1:n:100
        lrn.loss = mean(t[i:i+n-1])
        val = if i==1 lrn.loss else val*0.98 + lrn.loss*(1-0.98) end
        accumulate(met,lrn)
    end
    @test is_close(value(met), val)
end

@testset "AvgLoss" begin
    lrn = TestLearner()
    met = AvgLoss()
    reset(met)
    t = randn(100)
    n = 25
    for i in 1:n:100
        lrn.yb,lrn.loss = t[i:i+n-1],mean(t[i:i+n-1])
        accumulate(met,lrn)
    end
    @test is_close(value(met), mean(t))
end


"""
@testset "AccumMetric" begin
    #Go through a fake cycle with various batch sizes and computes the value of met
    x1,x2 = randn(20,5),torch.randn(20,5)
    Test = AccumMetric(_l2_mean)
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