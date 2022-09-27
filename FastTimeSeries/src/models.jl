
"""
blockmodel(inblock::TimeSeriesRow, outblock::OneHotTensor{0}, backbone)

Construct a model for time-series classification.
"""
function blockmodel(inblock::TimeSeriesRow,
                outblock::OneHotTensor{0}, 
                backbone)
    data   = zeros(Float32, inblock.nfeatures, 1, 1)
    output = backbone(data)
    Flux.reset!(backbone)
    return Models.RNNModel(backbone, outsize = length(outblock.classes), recout = size(output, 1))
end

"""
    blockbackbone(inblock::TimeSeriesRow)

Construct a recurrent backbone
"""
function blockbackbone(inblock::TimeSeriesRow)
    Models.StackedLSTM(inblock.nfeatures, 16, 10, 2);
end

# ## Tests

@testset "blockbackbone" begin 
    @test_nowarn FastAI.blockbackbone(TimeSeriesRow(1,140)) 
end