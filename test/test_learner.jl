using FastAI
using Flux: mse, @functor, Dense, Descent, train!, Params, gradient, update!
using Flux.Data: DataLoader
using Base: length, getindex

struct SynthDataset <: MapDataset
    x
    y
end

Base.getindex(md::SynthDataset,idx::Int) = (md.x[1,1,idx],md.y[1,1,idx])
Base.getindex(md::SynthDataset,rng::UnitRange) = (md.x[1,1,rng],md.y[1,1,rng])
Base.length(md::SynthDataset) = length(md.x)

"A simple DataBunch where `x` is random and `y = a*x + b` plus some noise."
function synth_dbunch(a=2, b=3, bs=16, n_train=10, n_valid=2)

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

function synth_learner(n_train=10, n_valid=2, cuda=false, lr=0.01)
    data = synth_dbunch() #n_train=n_train,n_valid=n_valid)
    return Learner(data, Dense(1,1), loss=mse, opt=Descent(0.001))
end

function one_batch(dl::DataLoader)
    for d in dl
        return d
    end
end

@testset "Learner" begin
    learn = synth_learner()
    add_cb!(learn,DummyCallback())
    
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