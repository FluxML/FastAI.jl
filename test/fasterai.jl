

@testset "FasterAI" begin
    @test length(listdatasources()) > 10

    @test !isempty(finddatasets(blocks=(Image, Label)))
    @test !isempty(finddatasets(blocks=(Image, LabelMulti)))
    @test !isempty(finddatasets(blocks=(Image, Mask)))

    @test ImageClassificationSingle ∈ findlearningtasks((Image, Label))
    @test ImageClassificationMulti ∈ findlearningtasks((Image, LabelMulti))
end
