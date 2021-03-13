# Status

## To-Dos

- [x] release DLPipelines.jl
    - update its documentation
    - register
- [x] release DataAugmentation.jl
    - add onehot-encoding of `MaskMulti`s
    - add tests for `PinOrigin`
    - fix bounds assignment for `PinOrigin` on `Keypoints`
- [x] move basic datasets and dataset utilities from DLDatasets.jl to FastAI.jl
    - [x] image classification datasets
    - [y] other datasets
- [x] add `ImageSegmentation` method
- training utilities
    - [x] `fitonecycle!`
    - [x] `finetune!`
    - [x] `lrfind!`
- wait for Flux@0.12.0 to be release
- plot recipes for learning rate finder, visualizing losses, hyperparameters
