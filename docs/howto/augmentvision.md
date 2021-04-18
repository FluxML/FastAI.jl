# How to augment vision data

Data augmentation is important to train models with good generalization ability, especially when the size of your dataset is limited. FastAI.jl gives you high-level helpers to use data augmentation in vision learning methods, but also allows directly using [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl), the underlying data augmentation library.

By default, the only augmentation that will be used in computer vision tasks, is a random crop, meaning that after images, keypoints and masks are resized to a similar size a random portion will be cropped during training. We can demonstrate this on the image segmentation task.

{cell=main}
```julia
using FastAI
using CairoMakie

dir = joinpath(datasetpath("dogscats"), "train")
data = loadtaskdata(dir, ImageClassificationTask)
classes = Datasets.getclassesclassification(dir)
method = ImageClassification(classes, (64, 128))

#=
traindl, valdl = methoddataloaders(data, method)
#xs, ys = FastAI.makebatch(method, data, [2, 2, 2, 2])
xs, ys = first(traindl)
plotbatch(method, xs, ys)
=#
```

{cell=main}
```julia
using MosaicViews
using DataAugmentation
p = FastAI.ProjectiveTransforms((64, 128), buffered = false)
image = getobs(data, 1).input
ims = [FastAI.run(p, Training(), image) for _ in 1:16]
mosaicview(ims; ncol = 4)

```

{cell=main}
```julia
tfm = Buffered(RandomResizeCrop((64, 128)))
ims = [apply(tfm, DataAugmentation.Image(image)) for _ in 1:16]
showgrid(ims, ncol = 4)
```