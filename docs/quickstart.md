# Quickstart

FastAI.jl makes it easy to train models for common tasks. For example, we can train an image classification model in just 6 lines.

```julia
using FastAI
using FastAI.Datasets
```

## Image classification

Train an image classifier from scratch:

```julia
data = Datasets.loadtaskdata(Datasets.datasetpath("imagenette2-160"), ImageClassificationTask)
method = ImageClassification(Datasets.getclassesclassification("imagenette2-160"), (160, 160))
learner = methodlearner(method, data, Models.xresnet18(), ToGPU(), Metrics(accuracy))
fitonecycle!(learner, 5)
```

Or finetune a pretrained model:

```julia
learner = methodlearner(method, data, Models.resnet50(pretrained=true))
finetune!(learner, 3)
```