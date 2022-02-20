# How to augment vision data

Data augmentation is important to train models with good generalization ability, especially when the size of your dataset is limited. FastAI.jl gives you high-level helpers to use data augmentation in vision learning methods, but also allows directly using [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl), the underlying data augmentation library.

By default, the only augmentation that will be used in computer vision tasks is a random crop, meaning that after images, keypoints and masks are resized to a similar size a random portion will be cropped during training. We can demonstrate this on the image classification task.

{cell=main output=false}
```julia
using FastAI
import FastAI: Image
import CairoMakie; CairoMakie.activate!(type="png")

data, blocks = loaddataset("imagenette2-160", (Image, Label))
method = BlockMethod(
    blocks,
    (
        ProjectiveTransforms((128, 128)),
        ImagePreprocessing(),
        OneHot()
    )
)
xs, ys = FastAI.makebatch(method, data, fill(4, 3))
showbatch(method, (xs, ys))     
```


Most learning methods let you pass additional augmentations as keyword arguments. For example, `ImageClassification` takes the `aug_projection` and `aug_image` arguments. FastAI.jl provides the [`augs_projection`](#) helper to quickly construct a set of projective data augmentations.

{cell=main}
```julia
method2 = BlockMethod(
    blocks,
    (
        ProjectiveTransforms((128, 128), augmentations=augs_projection()),
        ImagePreprocessing(),
        OneHot()
    )
)
xs2, ys2 = FastAI.makebatch(method2, data, fill(4, 3))
showbatch(method2, (xs2, ys2))
```


Likewise, there is an [`augs_lighting`](#) helper that adds contrast and brightness augmentation:

{cell=main}
```julia
method3 = BlockMethod(
    blocks,
    (
        ProjectiveTransforms((128, 128), augmentations=augs_projection()),
        ImagePreprocessing(augmentations=augs_lighting()),
        OneHot()
    )
)
xs3, ys3 = FastAI.makebatch(method3, data, fill(4, 3))
showbatch(method3, (xs3, ys3))
```
