# How to augment vision data

Data augmentation is important to train models with good generalization ability, especially when the size of your dataset is limited. FastAI.jl gives you high-level helpers to use data augmentation in vision learning methods, but also allows directly using [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl), the underlying data augmentation library.

By default, the only augmentation that will be used in computer vision tasks is a random crop, meaning that after images, keypoints and masks are resized to a similar size a random portion will be cropped during training. We can demonstrate this on the image classification task.
{cell=main output=false result=false style="display:none;"}
```julia
using Images: load
function showfig(f)
    save("fig.png", f)
    load("fig.png")
end
```

{cell=main, result=false}
```julia
using FastAI
using CairoMakie

dir = joinpath(datasetpath("dogscats"), "train")
data = loadtaskdata(dir, ImageClassificationTask)
classes = Datasets.getclassesclassification(dir)
method = ImageClassification(classes, (100, 128))
xs, ys = FastAI.makebatch(method, data, fill(4, 9))
f = FastAI.plotbatch(method, xs, ys)
```
{cell=main output=false style="display:none;"}
```julia
showfig(f)
```


Most learning methods let you pass additional augmentations as keyword arguments. For example, `ImageClassification` takes the `aug_projection` and `aug_image` arguments. FastAI.jl provides the [`augs_projection`](#) helper to quickly construct a set of projective data augmentations.

{cell=main result=false}
```julia
method2 = ImageClassification(classes, (100, 128), aug_projection=augs_projection())
xs2, ys2 = FastAI.makebatch(method2, data, fill(4, 9))
f = FastAI.plotbatch(method2, xs2, ys2)
```
{cell=main output=false style="display:none;"}
```julia
showfig(f)
```

Likewise, there is an [`augs_lighting`](#) helper that adds contrast and brightness augmentation:

{cell=main}
```julia
method3 = ImageClassification(
    classes, (100, 128),
    aug_projection=augs_projection(), aug_image=augs_lighting())
xs3, ys3 = FastAI.makebatch(method3, data, fill(4, 9))
f = FastAI.plotbatch(method3, xs3, ys3)
```