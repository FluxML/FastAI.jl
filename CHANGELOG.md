# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.5 (unreleased)

### Changed

- (BREAKING): Now uses [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) to create and load datasets and data containers
    - Replaces dependencies MLDataPattern.jl, LearnBase.jl, and DataLoaders.jl
    - Data containers must now implement the `Base.getindex`/`MLUtils.getobs` and `Base.length`/`MLUtils.numobs` interfaces.
    - Previously exported `MLDataPattern.datasubset` has been replaced by `MLUtils.ObsView`
    - Documentation has been updated appropriately
- (BREAKING): `FastAI.Vision` now lives in a separate package `FastVision` that holds all computer vision-related functionality. 
- (BREAKING): `FastAI.Tabular` now lives in a separate package `FastTabular` that holds all tabular data-related functionality. 

### Removed

- (BREAKING): `FastAI.Models` submodule. `Models` submodule of domain libraries, e.g. `FastVision.Models` should now be used.

## v0.4.3 (2022/05/14)

### Added 

- Feature registries let you find datasets, data recipes and learning tasks for your projects. It is now easier to search for functionality related to kinds of data and load it. See the updated [discovery tutorial](https://fluxml.ai/FastAI.jl/dev/i/?id=documents%2Fdocs%2Fdiscovery.md&id=references%2FFastAI.Registries.learningtasks)
- @Chandu-444 added first support for text datasets, adding the `Paragraph` block and `FastAI.Textual` submodule (https://github.com/FluxML/FastAI.jl/pull/207)

### Removed

- the old APIs for registries have been removed and functionality for accessing them (`finddatasets`, `loaddataset`) has been deprecated. See the updated docs for how to find functionality using the new feature registries.


## v0.4.2 (2022/04/30)

### Added

- Compatibility with FluxTraining.jl v0.3 (https://github.com/FluxML/FastAI.jl/pull/223)

## v0.4.1

### Added

- New documentation frontend based on Pollen.jl: https://fluxml.ai/FastAI.jl/dev/i/
- Now supports Flux.jl v0.13 (https://github.com/FluxML/FastAI.jl/pull/202)

### Changed

- Now has ImageIO.jl as a dependency to ensure that fast jpg loading using JpegTurbo.jl is used

## v0.4.0 (2022-03-19)

### Added

- Made block-based learning method more modular. `SupervisedMethod` now supplants `BlockMethod`.  [PR](https://github.com/FluxML/FastAI.jl/pull/188)
  - `getencodings` and `getblocks` should now be used to get block information and encodings from a method
  - See the [new tutorial training a Variational Autoencoder].
  - See also the docstrings for `AbstractBlockTask` and `SupervisedTask`

### Changed

- (BREAKING): all learning method names have been renamed to task, i.e `method*` -> `task*` and `Method*` -> `Task*`. Specifically, these exported symbols are affected:
  - `BlockMethod` -> `BlockTask`,
  - `describemethod` -> `describetask`,
  - `methodmodel` -> `taskmodel`,
  - `methoddataset` -> `taskdataset`,
  - `methoddataloaders` -> `taskdataloaders`,
  - `methodlossfn` -> `tasklossfn`,
  - `findlearningmethods` -> `findlearningtasks`,
  - `methodlearner` -> `tasklearner`,
  - `savemethodmodel` -> `savetaskmodel`,
  - `loadmethodmodel` -> `loadtaskmodel`
- `BlockMethod` now deprecated in favor of `SupervisedMethod`
- (INTERNAL) domain-specific functionality has moved to submodules `FastAI.Vision` (computer vision) and `FastAI.Tabular` (tabular data). Exports of `FastAI` are not affected.
- (INTERNAL) test suite now runs on InlineTest.jl

### Removed

## v0.3.0 (2021/12/11)

### Added

- A new API for visualizing data. See [this issue](https://github.com/FluxML/FastAI.jl/issues/154) for motivation. This includes:

    - High-level functions for visualizing data related to a learning method: `showsample`,  `showsamples`, `showencodedsample`, `showencodedsamples`, `showbatch`, `showprediction`, `showpredictions`, `showoutput`, `showoutputs`, `showoutputbatch`
    - Support for multiple backends, including a new text-based show backend that you can use to visualize data in a non-graphical environment. This is also the default unless `Makie` is imported.
    - Functions for showing blocks directly: `showblock`, `showblocks`
    - Interfaces for extension: `ShowBackend`, `showblock!`, `showblocks!`

### Removed

- The old visualization API incl. all its `plot*` methods: `plotbatch`, `plotsample`, `plotsamples`, `plotpredictions`


## 0.2.0 (2021/09/21)

### Added

- High-level API "FasterAI"
    - dataset recipes
    - learning method helpers
    - Find datasets and learning methods based on `Block`s: [`finddatasets`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.Datasets.Datasets.finddatasets.html), [`findlearningmethods`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.findlearningmethods.html)
    - [`loaddataset`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.Datasets.Datasets.loaddataset.html) for quickly loading data containers from configured recipes
- Data container recipes (`DatasetRecipe`, `loadrecipe`)
- Documentation setions for FasterAI interfaces:
    - [Discovery](https://fluxml.ai/FastAI.jl/dev/docs/discovery.md.html)
    - [Blocks and encodings](https://fluxml.ai/FastAI.jl/dev/docs/background/blocksencodings.md.html)
- New interfaces
    - `blockbackbone` creates a default backbone for an input block
- Support for tabular data along with recipes and learning methods:
    - [Tabular classification tutorial](https://fluxml.ai/FastAI.jl/dev/docs/notebooks/tabularclassification.ipynb.html)
    - [`TabularPreprocessing`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.TabularPreprocessing.html), [`TableRow`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.TableRow.html), [`TableDataset`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.Datasets.TableDataset.html), [`TabularClassificiationSingle`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.TabularClassificationSingle.html), [`TabularRegression`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.TabularRegression.html)


### Changed

- Documentation sections to reference FasterAI interfaces:
    - [README](https://fluxml.ai/FastAI.jl/dev/README.md.html)
    - [Introduction](https://fluxml.ai/FastAI.jl/dev/docs/introduction.md.html)
    - [Data containers](https://fluxml.ai/FastAI.jl/dev/docs/data_containers.md.html)
    - Combined how-tos on training [into a single page](https://fluxml.ai/FastAI.jl/dev/docs/notebooks/training.ipynb.html)
- Breaking changes to [`methodlearner`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.methodlearner.html):
    - now accepts `callbacks` as kwarg
    - `validdata` no longer keyword
    - `model` and `backbone` now kwargs; `isbackbone` removed. if neither `backbone` or `model` are given, uses `blockbackbone` for default backbone.
    - see updated docstring for details
