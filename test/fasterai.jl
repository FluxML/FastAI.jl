

@testset "FasterAI" begin
    @test length(listdatasources()) > 10

    @test length(finddatasets(blocks=(Image, Label))) > 0
    @test length(finddatasets(blocks=(Image, LabelMulti))) > 0
    @test length(finddatasets(blocks=(Image, Mask))) > 0

    @test ImageClassificationSingle ∈ findlearningtasks((Image, Label))
    @test ImageClassificationMulti ∈ findlearningtasks((Image, LabelMulti))
end
