"""
    TSRegressionSingle

Learning task for single-label time-series regression. Samples are standardized.
"""

function TSRegression(blocks::Tuple{<:TimeSeriesRow, <:Continuous}, data)
    return SupervisedTask(
        blocks,
        (
            setup(TSPreprocessing, blocks[1], data[1].table),
        ),
    )    
end

_tasks["tsregression"] = (
    id = "timeseries/regression",
    name = "Time-Series Regression",
    constructor = TSRegression,
    blocks = (TimeSeriesRow, Continuous),
    category = "supervised",
    description = """
        Time-Series regression task where every time-series has a single 
        regression label associated with it.
        """,
    package = @__MODULE__,
)