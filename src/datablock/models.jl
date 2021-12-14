


"""
    blockmodel(inblock, outblock[, backbone = blockbackbone(inblock)])

From a `backbone` model, construct a model suitable for learning
a mapping from `inblock` to `outblock`.
"""
function blockmodel end

blockmodel(inblock, outblock) = blockmodel(inblock, outblock, blockbackbone(inblock))

"""
    blockbackbone(inblock)

Create a default backbone that takes in block `inblock`.
"""
function blockbackbone end
