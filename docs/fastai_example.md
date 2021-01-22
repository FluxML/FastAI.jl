# Quickstart for fastai users

In the first chapter of the [fastai book](https://github.com/fastai/fastbook) the first code presented to the reader is this snippet:

```python
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```

Here is how you can do the same with FastAI.jl:

```julia
using FastAI

dataset = loaddataset(Pets)
method = ImageClassification(2, (224, 224))
dls = methoddataloaders(dataset, method)
model = methodmodel(method, backbone = Models.resnet34())

learner = Learner(model, dls, ADAM(), methodlossfn(method), Metrics(accuracy))
finetune!(learner, 1)
```




## What is missing?

- `Pets`: dataset needs to be added to *DLDatasets.jl*, basically same as `ImageNette/Woof`
- `methoddataloaders`: small wrapper around `MethodDataset` and `DataLoader`
- `resnet34`: which library to use that has pretrained weights
- `methodmodel`: put a classification head on a backbone model, calculating the sizes using `outputsize`
- `finetune!`: backbone freezing (parameter modification) and one-cycle learning rate scheduling (see [fastai implementation](https://github.com/fastai/fastai/blob/f2ab8ba78b63b2f4ebd64ea440b9886a2b9e7b6f/fastai/callback/schedule.py#L153))

Sketches:

- `methoddataloaders`

```julia
function methoddataloaders(datas::NTuple{2}, method; batchsize = 16)
    traindata, valdata = datas
    return (
        methoddataset(traindata, method, Training()),
        methoddataset(valdata, method, Validation()),
    )
end

methoddataloaders(data, method; pctgval = 0.2, kwargs...) =
    methoddataloaders(splitobs(data, at = pctgval); kwargs...)
```

- `methodmodel`

```julia
function methodmodel(method::ImageClassification, backbone)
    h, w, ch, b = outputsize(backbone, (method.sz..., 3, 1))
    return Chain(
        backbone,
        Chain(
            AdaptiveMeanPool((1,1)),
            flatten,
            Dense(ch, length(method.categories)),
        )
    )
end
```

- `finetune!`

```julia
function finetune!(learner, epochs; baselr = 2e-3, epochsfrozen = 1)
    e = length(learner.data.training)

    learner.model = freeze(learner.model, 1:1)
    setschedule!(learner, LearningRate => onecycle(
        e * epochsfrozen, baselr, start_ptcg = 0.99))
    fit!(learner, epochsfrozen)

    learner.model = unfreeze(learner.model)
    setschedule!(learner, LearningRate => onecycle(
        e * epochs,
        baselr / 2,
        start_val = baselr / 200))
    fit!(learner, epochs)

    return learner
end

# TODO: setschedule!, freeze
```