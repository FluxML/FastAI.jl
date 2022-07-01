"""
    TSClassificationSingle(blocks[, data])
Learning task for single-label time-series classification. Samples are normalized and 
classified into of the 'classes'.
"""
function TSClassificationSingle(
    blocks::Tuple{<:TimeSeriesRow, <:Label},
    data
)
    return SupervisedTask(
        blocks,
        (
            OneHot()
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