"""
    TSClassificationSingle(blocks[, data])

Learning task for single-label time-series classification. Samples are standardized 
and classified into one of `classes`.
"""

function TSClassificationSingle(blocks::Tuple{<:TimeSeriesRow,<:Label}, data)
    return SupervisedTask(
        blocks,
        (
            OneHot(),
            setup(TSPreprocessing, blocks[1], data[1].table)
        )
    )    
end

_tasks["tsclfsingle"] = (
    id = "timeseries/single",
    name = "Time-Series Classification (single-label)",
    constructor = TSClassificationSingle,
    blocks = (TimeSeriesRow, Label),
    category = "supervised",
    description = """
        Time-Series classification task where every time-series has a single 
        class label associated with it.
        """,
    package = @__MODULE__,
)