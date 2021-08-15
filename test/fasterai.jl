include("imports.jl")



@testset ExtendedTestSet "FasterAI" begin
    @test length(listdatasources()) > 10

    @test !isempty(finddatasets(blocks=(Image, Label)))
    @test !isempty(finddatasets(blocks=(Image, LabelMulti)))
    @test !isempty(finddatasets(blocks=(Image, Mask)))

    @test ImageClassificationSingle ∈ findlearningmethods((Image, Label))
    @test ImageClassificationMulti ∈ findlearningmethods((Image, LabelMulti))
end
