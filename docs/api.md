# FastAI.jl interfaces

## Training

### High-level

Quickly get started training and finetuning models using already implemented learning tasks and callbacks.

{.tight}
- [`tasklearner`](#)
- [`fit!`](#)
- [`fitonecycle!`](#)
- [`finetune!`](#)
- [`BlockTask`](#)
- callbacks

### Mid-level

{.tight}
- [`Learner`](#)
- [`taskmodel`](#)
- [`tasklossfn`](#)

### Low-level

{.tight}
- [`LearningTask`](#)
- [`encode`](#)
- [`encodeinput`](#)
- `decodey`

## Datasets

### High-level

Quickly download and load task data containers from the fastai dataset library.

{.tight}
- `load
- [`FastAI.Datasets.DATASETS`](#)

### Mid-level

Load and transform data containers.

{.tight}
- [`FastAI.Datasets.datasetpath`](#)
- [`FastAI.Datasets.FileDataset`](#)
- [`FastAI.Datasets.TableDataset`](#)
- [`mapobs`](#)
- [`groupobs`](#)
- [`joinobs`](#)
- [`groupobs`](#)

### Low-level

Full control over data containers.

{.tight}
- [`LearnBase.getobs`](#)
- [`LearnBase.nobs`](#)


