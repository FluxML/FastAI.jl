using FastAI
using Statistics
using Test
using Infiltrator

@testset "Metric" begin

    using FastAI: SmoothMetric, reset!, accumulate!, value, name

    @testset "SmoothMetric" begin
        met = SmoothMetric(0.98)
        reset!(met)
        t = randn(100)
        val = 0.0
        for (i,v) in enumerate(t)
            val = if i==1 v else val*0.98 + v*(1-0.98) end
            accumulate!(met,v)
        end
        @test is_close(value(met), val)
    end



    # @testset "AccumMetric" begin
    #     #Go through a fake cycle with various batch sizes and computes the value of met
    #     x1,x2 = randn(20,5),torch.randn(20,5)
    #     Test = AccumMetric(_l2_mean)
    #     @test compute_val(tst, x1, x2) ≈ _l2_mean(x1, x2)
    #     @test tst.preds == x1.view(-1)
    #     @test tst.targs == x2.view(-1)

    #     #test argmax
    #     x1,x2 = torch.randn(20,5),torch.randint(0, 5, (20,))
    #     tst = AccumMetric(_l2_mean, dim_argmax=-1)
    #     test_close(compute_val(tst, x1, x2), _l2_mean(x1.argmax(dim=-1), x2))

    #     #test thresh
    #     x1,x2 = torch.randn(20,5),torch.randint(0, 2, (20,5)).bool()
    #     tst = AccumMetric(_l2_mean, thresh=0.5)
    #     test_close(compute_val(tst, x1, x2), _l2_mean((x1 >= 0.5), x2))

    #     #test sigmoid
    #     x1,x2 = torch.randn(20,5),torch.randn(20,5)
    #     tst = AccumMetric(_l2_mean, activation=ActivationType.Sigmoid)
    #     test_close(compute_val(tst, x1, x2), _l2_mean(torch.sigmoid(x1), x2))

    #     #test to_np
    #     x1,x2 = torch.randn(20,5),torch.randn(20,5)
    #     tst = AccumMetric(lambda x,y: isinstance(x, np.ndarray) and isinstance(y, np.ndarray), to_np=True)
    #     assert compute_val(tst, x1, x2)

    #     #test invert_arg
    #     x1,x2 = torch.randn(20,5),torch.randn(20,5)
    #     tst = AccumMetric(lambda x,y: torch.sqrt(x.pow(2).mean()))
    #     test_close(compute_val(tst, x1, x2), torch.sqrt(x1.pow(2).mean()))
    #     tst = AccumMetric(lambda x,y: torch.sqrt(x.pow(2).mean()), invert_arg=True)
    #     test_close(compute_val(tst, x1, x2), torch.sqrt(x2.pow(2).mean()))
    # end
end