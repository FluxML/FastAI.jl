# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.4.0 (Unreleased)

### Added

### Changed

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
    - [Tabular classification tutorial](https://fluxml.ai/FastAI.jl/dev/notebooks/tabularclassification.ipynb.html)
    - [`TabularPreprocessing`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.TabularPreprocessing.html), [`TableRow`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.TableRow.html), [`TableDataset`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.Datasets.TableDataset.html), [`TabularClassificiationSingle`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.TabularClassificationSingle.html), [`TabularRegression`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.TabularRegression.html)


### Changed

- Documentation sections to reference FasterAI interfaces:
    - [README](https://fluxml.ai/FastAI.jl/dev/README.md.html)
    - [Introduction](https://fluxml.ai/FastAI.jl/dev/docs/introduction.md.html)
    - [Data containers](https://fluxml.ai/FastAI.jl/dev/docs/data_containers.md.html)
    - Combined how-tos on training [into a single page](https://fluxml.ai/FastAI.jl/dev/notebooks/training.ipynb.html)
- Breaking changes to [`methodlearner`](https://fluxml.ai/FastAI.jl/dev/REFERENCE/FastAI.methodlearner.html):
    - now accepts `callbacks` as kwarg
    - `validdata` no longer keyword
    - `model` and `backbone` now kwargs; `isbackbone` removed. if neither `backbone` or `model` are given, uses `blockbackbone` for default backbone.
    - see updated docstring for details
