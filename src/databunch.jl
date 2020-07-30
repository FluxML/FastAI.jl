
struct DataBunch
    train::Flux.Data.DataLoader
    valid::Flux.Data.DataLoader
end

train(db::DataBunch) = db.train
valid(db::DataBunch) = db.valid