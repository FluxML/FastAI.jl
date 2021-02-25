# FastAI.jl interfaces

## Training

### High-level

Quickly get started training and finetuning models using already implemented learning methods and callbacks.

- [`methodlearner`](#)
- [`fit!`](#)
- [`fitonecycle!`](#)
- [`finetune!`](#)
- `evaluate`
- learning methods
    - [`ImageClassification`](#)
    - [`ImageSegmentation`](#)
- callbacks

### Mid-level

- [`Learner`](#)
- [`methodmodel`](#)
- `adaptmodel`
- [`methodlossfn`](#)
- `Callback`

### Low-level

- [`LearningMethod`](#)
- [`LearningTask`](#)
- [`encode`](#)
- [`encodeinput`](#)
- `decodey`

## Datasets

### High-level

Quickly download and load task data containers from the fastai dataset library.

- [`Datasets.loadtaskdata`](#)
- [`Datasets.DATASETS`](#)

### Mid-level

Load and transform data containers.

- [`Datasets.datasetpath`](#)
- [`Datasets.FileDataset`](#)
- `Datasets.TableDataset`
- [`mapobs`](#)
- [`groupobs`](#)
- [`joinobs`](#)
- [`groupobs`](#)

### Low-level

Full control over data containers.

- [`getobs`](#)
- [`nobs`](#)


