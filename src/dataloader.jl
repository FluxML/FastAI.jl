#=

=#
struct DataLoader <: MapDataset
    ds::MapDataset
    bs::Int
    shuffle::Bool
    num_workers::Int
end

DataLoader(ds::MapDataset, bs=16, shuffle=true, num_workers=0) = DataLoader(ds,bs,shuffle,num_workers)

Base.getindex(dl::DataLoader,idx::Int) = getindex(dl.ds,idx)

Base.length(dl::DataLoader) = length(dl.ds) 

one_batch(dl::DataLoader) = dl.ds[1:dl.bs]


