using FastAI
using Flux: mse
using Base: length, getindex


struct SynthDataset <: MapDataset
    x
    y
end

Base.getindex(md::SynthDataset,idx::Int) = (md.x[idx],md.y[idx])
Base.getindex(sd::SynthDataset,rng::UnitRange) = [sd[i] for i in rng]
Base.length(md::SynthDataset) = length(x)

"A simple dataset where `x` is random and `y = a*x + b` plus some noise."
function synth_dbunch(a=2, b=3, bs=16, n_train=10, n_valid=2)

    function get_data(n)
        x = randn(bs*n)
        return SynthDataset(x, a*x .+ b + 0.1*randn(bs*n))
    end

    train_ds = get_data(n_train)
    valid_ds = get_data(n_valid)
    train_dl = DataLoader(train_ds) #, bs=bs, shuffle=true, num_workers=0)
    valid_dl = DataLoader(valid_ds) #, bs=bs, num_workers=0)
    return DataBunch(train_dl, valid_dl)
end

"A r"
mutable struct RegModel
    a
    b
end

RegModel() = RegModel(rand(),rand()) 

(m::RegModel)(x) = x*m.a .+ m.b

function synth_learner(n_train=10, n_valid=2, cuda=false, lr=0.01)
    data = synth_dbunch() #n_train=n_train,n_valid=n_valid)
    return Learner(data, RegModel(), loss_func=mse, lr=lr)
end

split(xy) = [x for (x,_) in xy],[y for (_,y) in xy]

@testset "Learner" begin
    learn = synth_learner()
    xb,yb = learn |> data_bunch |> train |> one_batch |> split
    pb = model(learn)(xb)
    init_loss = loss_func(learn)(xb, yb)
    fit!(learn,epoch_count=6)
    final_loss = loss(learn)
    @assert final_loss < init_loss
end