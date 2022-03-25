# Data containers

*This tutorial explains what data containers are, how they are used in FastAI.jl and how to create your own. You are encouraged to follow along in a REPL or a Jupyter notebook and explore the code. You will find small exercises at the end of some sections to deepen your understanding.*

## Introduction

In the [quickstart](quickstart.md) section, you have already come in contact with data containers. The following code was used to load a data container for image classification:

{cell=main, result=false, output=false style="display:none;"}
```julia
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
```
{cell=main, output=false}
```julia
using FastAI
import FastAI: Image
data, _ = loaddataset("imagenette2-160", (Image, Label))
```

A data container is any type that holds observations of data and allows us to load them with `getobs` and query the number of observations with `nobs`. In this case, each observation is a tuple of an image and the corresponding class; after all, we want to use it for image classification. 

{cell=main}
```julia
image, class = obs = getobs(data, 1)
@show class
image
```

{cell=main}
```julia
nobs(data)
```

[`loaddataset`](#) makes it easy to a load a data container that is compatible with some block types, but to get a better feel for what it does, let's look under the hood by creating the same data container using some mid-level APIs.

## Creating data containers from files

Before we recreate the data container, [`datasetpath`](#) downloads a dataset and returns the path to the extracted files.

{cell=main}
```julia
dir = datasetpath("imagenette2-160")
```

Now we'll start with [`FileDataset`](#) which creates a data container (here a `Vector`) of files given a path. We'll use the path of the downloaded dataset:

{cell=main}
```julia
files = FileDataset(dir)
```

`files` is a data container where each observation is a path to a file. We'll confirm that using `getobs`:


{cell=main}
```julia
p = getobs(files, 100)
```

Next we need to load an image and the corresponding class from the path. If you have a look at the folder structure of `dir` you can see that the parent folder of each file gives the name of class. So we can use the following function to load the `(image, class)` pair from a path:

{cell=main}
```julia
function loadimageclass(p)
    return (
        loadfile(p),
        pathname(pathparent(p)),
    )
end

image, class = loadimageclass(p)
@show class
image
```

Finally, we use [`mapobs`](#) to lazily transform each observation and have a data container ready to be used for training an image classifier.

{cell=main}
```julia
data = mapobs(loadimageclass, files);
```

### Exercises

1. Using [`mapobs`](#) and [`loadfile`](#), create a data container where every observation is only an image.
2. Change the above code to run on a different dataset from the list in `Datasets.DATASETS_IMAGECLASSIFICATION`.


## Splitting a data container into subsets

Until now, we've only created a single data container containing all observations in a dataset. In practice, though, you'll want to have at least a training and validation split. The easiest way to get these is to randomly split your data container into two parts. Here we split `data` into 80% training and 20% validation data. Note the use of [`shuffleobs`](#) to make sure each split has approximately the same class distribution.

{cell=main}
```julia
traindata, valdata = splitobs(shuffleobs(data), at = 0.8);
```

This is great for experimenting, but where possible you will want to use the official training/validation split for a dataset. Consider the image classification dataset folder structure:

```
- $dir
    - train
        - class1
            - image1.jpg
            - image2.jpg
            - ...
        - class2
        - ...
    - valid
        - class1
        - class2
        - ...
```

As you can see, the grandparent folder of each image indicates which split it is a part of. [`groupobs`](#) allows us to partition a data container using a function. Let's use it to split `filedata` based on the name of the grandparent directory. (We can't reuse `data` for this since it no longer carries the file information.)

{cell=main}
```julia
datagroups = groupobs(files) do p
    pathname(pathparent(pathparent(p)))  # equivalent to `grandparentname(p)`
end
trainfiles, validfiles = datagroups["train"], datagroups["val"]
```

Using this official split, it will be easier to compare the performance of your results with those of others'. 


## Dataset recipes

We saw above how different image classification datasets can be loaded with the same logic as long as they are in a common format. To encapsulate the logic for loading common dataset formats, FastAI.jl has `DatasetRecipe`s. When we used [`finddatasets`](#) in the [discovery tutorial](discovery.md), it returned pairs of a dataset name and a `DatasetRecipe`. For example, `"imagenette2-160"` has an associated [`ImageFolders`](#) recipe and we can load it using [`loadrecipe`] and the path to the downloaded dataset:

{cell=main}
```julia
name, recipe = finddatasets(blocks=(Image, Label), name="imagenette2-160")[1]
data, blocks = loadrecipe(recipe, datasetpath(name))
```

These recipes also take care of loading the data block information for the dataset. Read the [discovery tutorial](discovery.md) to find out more about that.