include("../imports.jl")

@testset ExtendedTestSet "TabularModel Components" begin
    @testset ExtendedTestSet "embeddingbackbone" begin
        embed_szs = [(5, 10), (100, 30), (2, 30)]
        embeds = Tabular.tabular_embedding_backbone(embed_szs, 0.)
        x = [rand(1:n) for (n, _) in embed_szs]

        @test size(embeds(x)) == (70, 1)
    end

    @testset ExtendedTestSet "continuousbackbone" begin
        n = 5
        contback = Tabular.tabular_continuous_backbone(n)
        x = rand(5, 1)
        @test size(contback(x)) == (5, 1)
    end

    @testset ExtendedTestSet "TabularModel" begin
        n = 5
        embed_szs = [(5, 10), (100, 30), (2, 30)]

        embeds = Tabular.tabular_embedding_backbone(embed_szs, 0.)
        contback = Tabular.tabular_continuous_backbone(n)

        x = ([rand(1:n) for (n, _) in embed_szs], rand(5, 1))

        tm = Models.TabularModel(embeds, contback; outsize=4)
        @test size(tm(x)) == (4, 1)

        tm2 = Models.TabularModel(embeds, contback, Chain(Dense(100, 4), x->Tabular.sigmoidrange(x, 2, 5)))
        y2 = tm2(x)
        @test all(y2.> 2) && all(y2.<5)

        cardinalities = [4, 99, 1]
        tm3 = Models.TabularModel(n, 4, [200, 100], cardinalities = cardinalities, size_overrides = (10, 30, 30))
        @test size(tm3(x)) == (4, 1)
    end
end
