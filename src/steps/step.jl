abstract type PipelineStep end


"""
    run(step::PipelineStep, context, data)

Applies the operation `step` to `data`
"""
function run end

"""
    run!(buf, step::PipelineStep, context, data)

Applies the operation `step` inplace to `buf`. `buf` is mutated.
"""
function run! end


"""
    invert(step::PipelineStep, context, data)

Applies the inverse of the operation `step` to `data`
"""
function invert end


"""
    invert!(buf, step::PipelineStep, context, data)

Applies the inverse of the operation `step` to `buf` inplace. `buf` is mutated,
"""
function invert! end
