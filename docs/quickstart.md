# Quickstart

FastAI.jl makes it easy to train models for common tasks. For example, we can train an image classification model in just 6 lines.

```julia
using FastAI
```

## Image classification

```julia
dataset = loaddataset("imagenette2-160")
method = ImageClassification(loadclasses("imagenette2-160"), (224, 224))
dls = methoddataloaders(dataset, method)
model = methodmodel(method, Models.xresnet18());

learner = Learner(model, dls, ADAM(), methodlossfn(method), ToGPU(), Metrics(accuracy))
fitonecycle!(learner, 5)
```