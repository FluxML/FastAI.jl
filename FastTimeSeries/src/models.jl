
"""
blockmodel(inblock::TimeSeriesRow, outblock::OneHotTensor{0}, backbone)

Construct a model for time-series classification.
"""
function blockmodel(inblock::TimeSeriesRow,
                outblock::OneHotTensor{0}, 
                backbone)
    #TODO: Use Flux.outputsize here.
    data   = rand(Float32, inblock.nfeatures, 32, inblock.obslength)
    # data   = [rand(Float32, inblock.nfeatures, 32) for _ ∈ 1:inblock.obslength]
    output = backbone(data)
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

@testset "blockbackbone" begin @test_nowarn FastAI.blockbackbone(TimeSeriesRow(1,140)) end