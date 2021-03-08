# Status

## To-Dos

- release DLPipelines.jl
    - update its documentation
    - register
- release DataAugmentation.jl
    - add onehot-encoding of `MaskMulti`s
    - add tests for `PinOrigin`
    - fix bounds assignment for `PinOrigin` on `Keypoints`
- move basic datasets and dataset utilities from DLDatasets.jl to FastAI.jl
    - [x] image classification datasets
    - [ ] other datasets
- add `ImageSegmentation` method
- training utilities
    - [x] `fitonecycle!`
    - [ ] `finetune!`
    - [ ] `lrfind!`
- wait for Flux@0.12.0 to be released
- plot recipes for learning rate finder, visualizing losses, hyperparameters
