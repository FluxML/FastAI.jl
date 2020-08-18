# PyTorch Core Modules

The following modules are PyTorch constructs that fastai builds on, but they don't exist in Julia and Flux.

## Dataset

A dataset can either be an [`IterableDataset`](@ref) or a [`MapDataset`](@ref). If you want to create a custom dataset, then you should implement one of these two interfaces.

```@docs
IterableDataset
MapDataset
```

There are also several dataset types that represent combinations of datasets. If `ds1`, `ds2`, ..., `dsn` are the same kind of dataset, then `ds1 ++ ds2 ++ ... ++ dsn` will be a combined dataset of that sort. If [`IterableDataset`](@ref)s and [`MapDataset`](@ref)s are combined, the result will be an `IterableDataset`. This is useful to assemble different existing dataset streams. The chainning operation is done on-the-fly, so concatenating large-scale [`IterableDataset`](@ref)s with this type will be efficient.

```@docs
ConcatDataset
ChainDataset
++
```

Lastly, we can also take random and fixed subsets of a [`MapDataset`](@ref).

```@docs
SubsetDataset
subset
random_split
```