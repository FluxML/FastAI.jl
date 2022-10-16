# How to find functionality

For some kinds of functionality, FastAI.jl provides feature registries that allow you to search for and use features. The following registries currently exist:

- [`datasets`](#) to download and unpack datasets,
- [`datarecipes`](#) to load datasets into [data containers](/doc/docs/data_containers.md) that are compatible with a learning task; and
- [`learningtasks`](#) to find learning tasks that are compatible with a dataset

!!! note "Domain packages"

    Functionality is registered by domain packages such as [`FastVision`](#) and [`FastTabular`](#). You need to import the respective packages to be able to find their functionality in their registry.

To load functionality:

1. Get an entry using its ID
    {cell}
    ```julia
    using FastAI, FastVision
    entry = datasets()["mnist_var_size_tiny"]
    ```
2. And load it
    {cell}
    ```julia
    load(entry)
    ```


## Datasets

{cell}
```julia
using FastAI
datasets()
```

## Data recipes

{cell}
```julia
using FastAI
datarecipes()
```

## Learning tasks

{cell}
```julia
using FastAI
learningtasks()
```
