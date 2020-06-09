using Random: randperm

"""
Types representing the concept `Dataset`.  In Julia duck typing, implementing an
interface just requires implementing a set of required fuctions. 

There are two kinds of Dataset: IterableDataset and MapDataset

IterableDataset

Every Dataset is an IterableDataset and is a iterable of data samples.
IterableDataset is particularly useful when data come from a stream.

Required functions are:

iterate(it<:IterableDataset)		    Returns either a tuple of the first item and initial state or nothing if empty
iterate(it<:IterableDataset, state)		Returns either a tuple of the next item and next state or nothing if no items remain

See Julia iteration documentation for more information

    https://docs.julialang.org/en/v1/manual/interfaces/



Some Datasets are also MapDataset that map integer ids to data samples.  
All MapDatasets are also IterableDatasets  
    
Required functions:

    (md<:MapDataset)(idx::Int) a MapDataset is a callable type that maps an integer id 
    to a sample.  Legal IDs are between 1 and the length of this dataset.

    length(md<:MapDataset) returns the number of samples in this MapDataset.  Used
    by many `Sampler` implementations and the default options of `DataLoader`.

TODO later...
`DataLoader` by default constructs a index
sampler that yields integral indices.  To make it work with a map-style
dataset with non-integral indices/keys, a custom sampler must be provided.

"""

abstract type IterableDataset end
abstract type MapDataset <: IterableDataset end

struct ConcatDataset <: MapDataset
    ds1:: MapDataset
    ds2:: MapDataset
end

Base.length(cd::ConcatDataset) = length(cd.ds1) + length(cd.ds2)
 
(cd::ConcatDataset)(idx::Int) = idx > length(cd.ds1) ? cd.ds2(idx-length(cd.ds1)) : cd.ds1(idx)

struct ChainDataset <: IterableDataset
    ds1:: IterableDataset
    ds2:: IterableDataset
end

function iterate(cd::ChainDataset)
    it = iterate(cd.ds1)
    return isnothing(it) ? iterate(cd.ds2) : it
end		    

function iterate(cd::ChainDataset, state)	
    it = iterate(cd.ds1,state)	
    return isnothing(it) ? iterate(cd.ds2,state) : it
end	


"""
Concatenate multiple MapDatasets.  Given ds1, ds2... dsn are the same kind of 
dataset ds1 ++ ds2 ++ ... ++ dsn will be a combined dataset of that sort

This is useful to assemble different existing dataset streams.
The chainning operation is done on-the-fly, so concatenating large-scale
IterableDatasets with this class will be efficient.
"""
++(ds1:: MapDataset,ds2:: MapDataset) = ConcatDataset(ds1,ds2)
++(ds1:: IterableDataset,ds2:: IterableDataset) = ChainDataset(ds1,ds2)
++(ds1:: IterableDataset,ds2:: IterableDataset, ds3...) = foldl(++, (ds2 , ds3...), init=a)

struct SubsetDataset <: MapDataset
    ds:: MapDataset
    idxs:: Array{Int}
end

iterate(sd::SubsetDataset) = length(sd)>0 ? (sd.dataset(sd.idxs[1],2)) : nothing	    

iterate(sd::SubsetDataset, state) = state[2] <= length(sd) ? (sd(sd.idxs[state[2]]), state[2]+1) : nothing	

Base.length(sd::SubsetDataset) = length(sd.indices)
 
(sd::SubsetDataset)(idx::Int) = sd.ds(sd.idxs[idx])

"""
Subset of a MapDataset at specified indices.

Arguments:
    dataset: The whole Dataset
    indices: Indices in the whole set selected for subset
"""
subset(dataset:: MapDataset, indices:: Array{Int}) = SubsetDataset(dataset,indices)


"""
Randomly split a dataset into non-overlapping new datasets of given lengths.

Arguments:
    dataset: Dataset to be split
    lengths: lengths of splits to be produced
"""
function random_split(dataset:: MapDataset, lengths:: Array{Int})
    @assert sum(lengths) > length(dataset), "Sum of split lengths may not be greater than the length of the input dataset!"
    indices = randperm(sum(lengths))
    start = 1
    s = []
    for l in lengths
        push!(s,subset(dataset, indices[start:start+l-1]))
        start += l
    end
    return s
end

"""
TODO later...

When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
iterator. When :attr:`num_workers > 0`, each worker process will have a
different copy of the dataset object, so it is often desired to configure
each copy independently to avoid having duplicate data returned from the
workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
process, returns information about the worker. It can be used in either the
dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
:attr:`worker_init_fn` option to modify each copy's behavior.

Example 1: splitting workload across all workers in :meth:`__iter__`::

    >>> class MyIterableDataset(torch.utils.data.IterableDataset):
    ...     def __init__(self, start, end):
    ...         super(MyIterableDataset).__init__()
    ...         assert end > start, "this example code only works with end >= start"
    ...         self.start = start
    ...         self.end = end
    ...
    ...     def __iter__(self):
    ...         worker_info = torch.utils.data.get_worker_info()
    ...         if worker_info is None:  # single-process data loading, return the full iterator
    ...             iter_start = self.start
    ...             iter_end = self.end
    ...         else:  # in a worker process
    ...             # split workload
    ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
    ...             worker_id = worker_info.id
    ...             iter_start = self.start + worker_id * per_worker
    ...             iter_end = min(iter_start + per_worker, self.end)
    ...         return iter(range(iter_start, iter_end))
    ...
    >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
    >>> ds = MyIterableDataset(start=3, end=7)

    >>> # Single-process loading
    >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
    [3, 4, 5, 6]

    >>> # Mult-process loading with two worker processes
    >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
    >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
    [3, 5, 4, 6]

    >>> # With even more workers
    >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20)))
    [3, 4, 5, 6]

Example 2: splitting workload across all workers using :attr:`worker_init_fn`::

    >>> class MyIterableDataset(torch.utils.data.IterableDataset):
    ...     def __init__(self, start, end):
    ...         super(MyIterableDataset).__init__()
    ...         assert end > start, "this example code only works with end >= start"
    ...         self.start = start
    ...         self.end = end
    ...
    ...     def __iter__(self):
    ...         return iter(range(self.start, self.end))
    ...
    >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
    >>> ds = MyIterableDataset(start=3, end=7)

    >>> # Single-process loading
    >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
    [3, 4, 5, 6]
    >>>
    >>> # Directly doing multi-process loading yields duplicate data
    >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
    [3, 3, 4, 4, 5, 5, 6, 6]

    >>> # Define a `worker_init_fn` that configures each dataset copy differently
    >>> def worker_init_fn(worker_id):
    ...     worker_info = torch.utils.data.get_worker_info()
    ...     dataset = worker_info.dataset  # the dataset copy in this worker process
    ...     overall_start = dataset.start
    ...     overall_end = dataset.end
    ...     # configure the dataset to only process the split workload
    ...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    ...     worker_id = worker_info.id
    ...     dataset.start = overall_start + worker_id * per_worker
    ...     dataset.end = min(dataset.start + per_worker, overall_end)
    ...

    >>> # Mult-process loading with the custom `worker_init_fn`
    >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
    >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
    [3, 5, 4, 6]

    >>> # With even more workers
    >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
    [3, 4, 5, 6]
"""
