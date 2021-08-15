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

### Changed