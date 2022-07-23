
"""
blockmodel(inblock::TimeSeriesRow, outblock::OneHotTensor{0}, backbone)

Construct a model for time-series classification.
"""
function blockmodel(inblock::TimeSeriesRow,
                outblock::OneHotTensor{0}, 
                backbone)
    data   = [rand(Float32, inblock.nfeatures, 32) for _ ∈ 1:inblock.obslength]
    output = backbone(data)
    outs   = size(output)[1]
    return Models.RNNModel(backbone, outsize = length(outblock.classes), recout = outs)
end

"""
    blockbackbone(inblock::TimeSeriesRow)

Construct a recurrent backbone
"""
function blockbackbone(inblock::TimeSeriesRow)
    Models.StackedLSTM(inblock.nfeatures, 16, 10, 2);
end

# ## Tests

@testset "blockbackbone" begin @test_nowarn FastAI.blockbackbone(TimeSeriesRow(1,140)) end