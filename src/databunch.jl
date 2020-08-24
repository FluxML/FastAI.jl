"""
    DataBunch
    DataBunch(train::Flux.Data.DataLoader, valid::Flux.Data.DataLoader)

A `DataBunch` is a bunched `train` and `valid` (validation) dataloader.
"""
struct DataBunch
    train::Flux.Data.DataLoader
    valid::Flux.Data.DataLoader
end

"""
    train(db::DataBunch)

Get the `db.train` dataloader.
"""
train(db::DataBunch) = db.train
"""
    valid(db::DataBunch)

Get the `db.valid` (validation) dataloader.
"""
valid(db::DataBunch) = db.valid