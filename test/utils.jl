using FastAI
using Flux: mse, @functor, Dense, Descent, train!, Params, gradient, update!
using Flux.Data: DataLoader

"A simple DataBunch where `x` is random and `y = a*x + b` plus some noise."
function test_dbunch(a=2, b=3, bs=16, n_train=10, n_valid=2)

    function get_data(n)
        xy = [] 
        for i in 1:bs*n  
            x = rand()
            y = a*x .+ b + 0.1*rand()
            push!(xy,([x],[y]))
        end
        return DataLoader(xy, batchsize=bs, shuffle=true) 
    end

    train_dl = get_data(n_train)
    valid_dl = get_data(n_valid)
    return DataBunch(train_dl, valid_dl)
end

struct TestCallback <: AbstractCallback end

function test_learner(n_train=10, n_valid=2, cuda=false, lr=0.01)
    data = test_dbunch() #n_train=n_train,n_valid=n_valid)
    return Learner(data, Dense(1,1), loss=mse, opt=Descent(0.001))
end

function one_batch(dl::DataLoader)
    for d in dl
        return d
    end
end

function is_close(a,b;eps=1e-5)
    #@show a
    #@show b
    return (a-b)^2 < eps^2
end

