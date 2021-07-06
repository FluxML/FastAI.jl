struct TabularTransforms <: PipelineStep
    tfms
end

function run(tt::TabularTransforms, _, sample)
    DataAugmentation.apply(tt.tfms, sample)
end

"""
The helper functions defined below can be used for quickly constructing a dictionary,
which will be required for creating various tabular transformations available in DataAugmentation.jl.

These functions assume that the table in the TableDataset object td has Tables.jl columnaccess interface defined.
"""

function gettransformationdict(td, ::Type{DataAugmentation.NormalizeRow}, cols)
    dict = Dict()
    for col in cols
        vals = skipmissing(Tables.getcolumn(td.table, col))
        dict[col] = (Statistics.mean(vals), Statistics.std(vals))
    end
    dict
end

function gettransformationdict(td, ::Type{DataAugmentation.FillMissing}, cols)
    dict = Dict()
    for col in cols
        vals = skipmissing(Tables.getcolumn(td.table, col))
        dict[col] = Statistics.median(vals)
    end
    dict
end

function gettransformationdict(td, ::Type{DataAugmentation.Categorify}, cols)
    dict = Dict()
    for col in cols
        vals = Tables.getcolumn(td.table, col)
        dict[col] = unique(vals)
    end
    dict
end
        