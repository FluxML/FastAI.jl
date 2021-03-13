# Data containers

*This tutorial explains what data containers are, how they are used in FastAI.jl and how to create your own. You are encouraged to follow along in a REPL or a Jupyter notebook and explore the code. You will find small exercises at the end of some sections to deepen your understanding.*

## Introduction

In the [quickstart](quickstart.md) section, you have already come in contact with data containers. The following code was used to load a data container for image classification:

{cell=main, result=false, output=false style="display:none;"}
```julia
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
```
{cell=main}
```julia
using FastAI
using FastAI.Datasets
using FastAI.Datasets: datasetpath, loadtaskdata

NAME = "imagenette2-160"
dir = datasetpath(NAME)
data = loadtaskdata(dir, ImageClassificationTask)
```

A data container is any type that holds observations of data and allows us to load them with `getobs` and query the number of observations with `nobs`:

{cell=main}
```julia
obs = getobs(data, 1)
```

{cell=main}
```julia
nobs(data)
```

In this case, each observation is a tuple of an image and the corresponding class; after all, we want to use it for image classification. 

{cell=main}
```julia
image, class = obs
@show class
image
```

As you saw above, the `Datasets` submodule provides functions for loading and creating data containers. We used [`Datasets.datasetpath`](#) to download a dataset if it wasn't yet and get the folder it was downloaded to. Then, [`Datasets.loadtaskdata`](#) took the folder and loaded a data container suitable for image classification. FastAI.jl makes it easy to download the datasets from fastai's collection on AWS Open Datasets. For the full list, see [`Datasets.DATASETS`](#)


### Exercises

1. Have a look at the other image classification datasets in [`Datasets.DATASETS_IMAGECLASSIFICATION`](#) and change the above code to load a different dataset.


## Creating data containers from files

`loadtaskdata` makes it easy to get started when your dataset already comes in the correct format, but alas, datasets come in all different shapes and sizes. Let's create the same data container, but now using more general functions FastAI.jl provides to get a look behind the scenes. If each observation in your dataset is a file in a folder, [`FileDataset`](#) conveniently creates a data container given a path. We'll use the path of the downloaded dataset:

{cell=main}
```julia
using FastAI.Datasets: FileDataset

filedata = FileDataset(dir)
```

`filedata` is a data container where each observation is a path to a file. We'll confirm that using `getobs`:


{cell=main}
```julia
p = getobs(filedata, 100)
```

Next we need to load an image and the corresponding class from the path. If you have a look at the folder structure of `dir` you can see that the parent folder of each file gives the name of class. So we can use the following function to load the `(image, class)` pair from a path:

{cell=main}
```julia
using FastAI.Datasets: loadfile, filename

function loadimageclass(p)
    return (
        Datasets.loadfile(p),
        filename(parent(p)),
    )
end

image, class = loadimageclass(p)
@show class
image
```

Finally, we use [`mapobs`](#) to lazily transform each observation and have a data container ready to be used for training an image classifier.

{cell=main}
```julia
data = mapobs(loadimageclass, filedata);
```

### Exercises

1. Using `mapobs` and `loadfile`, create a data container where every observation is only an image.


## Splitting a data container into subsets

Until now, we've only created a single data container containing all observations in a dataset. In practice, though, you'll want to have at least a training and validation split. The easiest way to get these is to randomly split your data container into two parts. Here we split `data` into 80% training and 20% validation data. Note the use of `shuffleobs` to make sure each split has approximately the same class distribution.

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
trainfiledata, validfiledata = groupobs(filedata) do p
    filename(parent(parent(p)))
end
nobs(trainfiledata), nobs(validfiledata)
```

Using this official split, it will be easier to compare the performance of your results with those of others'. 
