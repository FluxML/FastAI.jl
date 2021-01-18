# Getting started

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
dls = MethodDataLoaders(dataset, method)
learner = Learner(resnet34(), dls, ADAM(), logitcrossentropy)
finetune!(learner, 1)
```