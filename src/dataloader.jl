#=


struct DataLoader <: MapDataset
    ds::MapDataset
    bs::Int
    shuffle::Bool
    num_workers::Int
end

DataLoader(ds::MapDataset, bs=16, shuffle=true, num_workers=0) = DataLoader(ds,bs,shuffle,num_workers)

Base.getindex(dl::DataLoader,idx::Int) = getindex(dl.ds,idx)

Base.length(dl::DataLoader) = length(dl.ds) 

one_batch(dl::DataLoader) = dl.bs >= length(dl.ds) ? dl.ds[1:end] : dl.ds[1:dl.bs]

Base.iterate(dl::DataLoader, offset=1) = offset >= length(dl.ds) ? nothing : offset+dl.bs >= length(dl.ds) ? (dl.ds[offset:end],offset+dl.bs) : (dl.ds[offset:offset+dl.bs-1],offset+dl.bs)		    
=#

