# Image classification

When doing image classification, we want to train a model to classify a given image into one or more classes.

## Single-label classification

In the simple case, every image will have one class from a list associated with it. For example, the Cats&Dogs dataset contains pictures of cats and dogs. The learning method [`ImageClassification`](#) handles single-label image classification. Let's load some samples and visualize them:

{cell=main output=false result=false style="display:none;"}
```julia
using Images: load
function showfig(f)
    save("fig.png", f)
    load("fig.png")
end
```
{cell=main result=false output=false}
```julia
using CairoMakie
using FastAI
dir = joinpath(datasetpath("dogscats"), "train")
data = loadtaskdata(dir, ImageClassification)
samples = [getobs(data, i) for i in rand(1:nobs(data), 9)]
classes = Datasets.getclassesclassification(dir)
method = ImageClassification(classes, (128, 128))
f = plotsamples(method, samples)
```
{cell=main output=false style="display:none;"}
```julia
showfig(f)
```

With a method and a data container, we can easily construct a [`Learner`](#):

{cell=main}
```julia
learner = methodlearner(method, shuffleobs(data), Models.xresnet18())
```

If we have a look at a training batch, we can see that the images are resized and cropped to the same size:

{cell=main}
```julia
(xs, ys), _ = iterate(learner.data.training)
plotbatch(method, xs, ys)
```
