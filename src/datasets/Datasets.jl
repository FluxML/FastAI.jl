"""
    module Datasets

Commonly used datasets and utilities for creating data containers.

ToDos:

- add localization/segmentation datasets
- add labels for classification datasets

## Interface

DC<D> is a data container with observations of type D (i.e. `typeof(getobs(::DC<D>, i)::D)`)

Transformations:

- `mapobs(f::(D -> E), ::DC<D>)::DC<E>`

    Map a function (or a tuple of functions) over a data container.

- `Tuple(DC<D1>, ..., DC<DN>)::DC<(D1,...,DN)>`

    Combine multiple data containers into a single data container that returns tuples
    of the each's observations.

- `filterobs(f, DC<D>)::DC<D>`

    Keep only observations for which `f(obs) === true`.

- `groupobs(f, DC<D>)::(DC1<D>, ..., DCN<D>)` with `N` the unique return values of `f(::D)`

- `joinobs(f, DC<D1>, ..., DC<DN>)::DC<D>`

    Combines N datasets into a single one, "concatenating" them.

Primitive datasets:

- `FileDataset(dir; filterfn)`

    Each file in `dir` is one observation. Currently implemented in DLDatasets.jl with
    FileTrees.jl and observation type `FileTrees.File`.

- `TableDataset(table)`

    Every row in the table is an observation. Could use Tables.jl interface to be compatible
    with tons of packages.

## Examples

Loading and splitting an image classification dataset stored in the
same file structure as ImageNette, i.e.:

- train
    - class1
        - obs1
        - ...
        - obs2
    - class1
        - obs1
        - ...
        - obs2
- valid
    - ...

```julia
# file dataset of images `ds::DC<FileTrees.File>`
ds = FileDataset(DIR; filterfn = file -> extension(file) == "jpg")

# split into train and validation based on grandparent directory
trainds, valds = groupobs(file -> file.parent.parent.name == "train", ds)

# map (file -> input, file -> label) functions over containers to transform to type DC<(image, label)>
trainds = mapobs((FileIO.load, file -> file.parent.name), trainds)
# which is shorthand for
trainds = (
    mapobs(FileIO.load, trainds),
    mapobs(file -> file.parent.name, trainds),
)
```
---

Turning a container of (input, target) into a container of (x, y) and then an iterator
of batches (xs, ys). This is pretty much all `methoddataset` and `methoddataloaders` do.

```julia
# ds of (image, label) for example from above example
ds = ...
method = ImageClassification(...)

xyds = mapobs(ds) do (image, label)
    return encode(method, Training(), (image, label))
end

# data iterator ready to be used in training loop
dl = DataLoader(xyds, 16)

```

---

Loading an image dataset without labels for inference.

```julia
# ds contains original size images
ds = mapobs(FileIO.load(filterobs(file -> extension(file) == "jpg", FileDataset(DIR)))

transform = ProjectiveTransforms((128, 128))
ds = mapobs(ds) do obs
    run(transform, Validation(), obs)
end
# Now each observation is a center-cropped image of size (128, 128)
```
"""
module Datasets


using ..FastAI

using DataDeps
using FilePathsBase
using FilePathsBase: filename
import FileIO
using FileTrees
using MLDataPattern
using MLDataPattern: splitobs
import LearnBase
using Colors
using FixedPointNumbers

include("fastaidatasets.jl")

function __init__()
    initdatadeps()
end

include("containers.jl")
include("transformations.jl")

include("load.jl")


export
    # reexports from MLDataPattern
    splitobs,

    # container transformations
    mapobs,
    filterobs,
    groupobs,
    joinobs,

    # primitive containers
    FileDataset,
    TableDataset,
    Tokenizer,

    # utilities
    isimagefile,
    loadfile,
    filename,
    tokenize,

    # datasets
    DATASETS,
    loadtaskdata,
    datasetpath

end  # module
