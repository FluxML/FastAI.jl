{cell=main}
```julia
using FastAI
using FastAI.Datasets
using CairoMakie
```

{cell=main}
```julia
taskdata = Datasets.loadtaskdata(Datasets.datasetpath("camvid_tiny"), FastAI.ImageSegmentationTask);
image, mask = getobs(taskdata, 1)
summary.((image, mask))
```

{cell=main}
```julia
method = ImageSegmentation(Datasets.getclassessegmentation("camvid_tiny"), (96, 128));
```
{cell=main}
```julia
traindl, valdl = methoddataloaders(taskdata, method, 4);
xs, ys = batch = first(traindl)
summary.((xs, ys))
```
{cell=main}
```julia
FastAI.plotbatch(method, xs, ys)
```