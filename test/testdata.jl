using Flux

# test model with linear coefficient
struct TestModel
    coeff
end
TestModel(coeff::Number) = TestModel([coeff])
Flux.trainable(m::TestModel) = (m.coeff,)
(m::TestModel)(x) = x .* m.coeff
Flux.@functor TestModel


# test data

function testbatch(batchsize, coeff)
    xs = rand(1.:100., batchsize)
    return (xs, xs .* coeff)
end


function testbatches(n::Int, coeff, batchsize = 8)
    (testbatch(batchsize, coeff) for _ ∈ 1:n)
end



function testlearner(args...; nbatches = 16, coeff = 3, batchsize = 8, kwargs...)
    model = TestModel(rand())
    data = collect(testbatches(nbatches, coeff, batchsize))
    opt = Descent(0.001)
    Learner(
        model,
        (data, data),
        opt,
        Flux.mae,
        args...;
        usedefaultcallbacks = false,
        kwargs...)
end
