using Flux.Data: DataLoader

struct DataBunch
    train::DataLoader
    valid::DataLoader
end

train(db::DataBunch) = db.train
valid(db::DataBunch) = db.valid