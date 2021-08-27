# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2020-07-32 – now

### Added
- High-level API "FasterAI"
    - Find datasets and learning methods based on `Block`s: `finddataset`, `findlearningmethods`
    - `loaddataset` for quickly loading data containers from configured recipes
- Data container recipes (`DatasetRecipe`, `loadrecipe`)
- Documentation setions for FasterAI interfaces:
    - [Discovery](https://fluxml.ai/FastAI.jl/dev/docs/discovery.md.html)
    - [Blocks and encodings](https://fluxml.ai/FastAI.jl/dev/docs/background/blocksencodings.md.html)



### Changed

- Documentation sections to reference FasterAI interfaces:
    - [README](https://fluxml.ai/FastAI.jl/dev/README.md.html)
    - [Introduction](https://fluxml.ai/FastAI.jl/dev/docs/introduction.md.html)
    - [Data containers](https://fluxml.ai/FastAI.jl/dev/docs/data_containers.md.html)
    - Combined how-tos on training [into a single page](https://fluxml.ai/FastAI.jl/dev/notebooks/training.ipynb.html)