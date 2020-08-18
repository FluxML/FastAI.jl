# Basic Data

Basic types to contain the data for model training. This module defines the basic [`DataBunch`](@ref) struct that is used inside [`Learner`](@ref) to train a model. The fields are generic and can take any kind of fastai dataset (see [`IterableDataset`](@ref) and [`MapDataset`](@ref)) or [Flux.Data.DataLoader](https://fluxml.ai/Flux.jl/stable/data/dataloader/#Flux.Data.DataLoader). You'll find helpful functions in the data module of every application to directly create this DataBunch for you.

```@docs
DataBunch
```