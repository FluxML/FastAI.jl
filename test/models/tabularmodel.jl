include("../imports.jl")

@testset ExtendedTestSet "TabularModel Components" begin
    @testset ExtendedTestSet "embeddingbackbone" begin
        embed_szs = [(5, 10), (100, 30), (2, 30)]
        embeds = embeddingbackbone(embed_szs, 0.)
        x = [rand(1:n) for (n, _) in embed_szs]

        @test size(embeds(x)) == (70, 1)
    end

    @testset ExtendedTestSet "continuousbackbone" begin
        n = 5
        contback = continuousbackbone(n)
        x = rand(5, 1)
        @test size(contback(x)) == (5, 1)
    end

    @testset ExtendedTestSet "TabularModel" begin
        n = 5
        embed_szs = [(5, 10), (100, 30), (2, 30)]
        
        embeds = embeddingbackbone(embed_szs, 0.)
        contback = continuousbackbone(n)

        tm = TabularModel(
            embeds, 
            contback, 
            [200, 100],
            n_cat=3,
            n_cont=5,
            out_sz=4
        )
        x = ([rand(1:n) for (n, _) in embed_szs], rand(5, 1))
        @test size(tm(x)) == (4, 1)

        tm2 = TabularModel(
            embeds, 
            contback, 
            [200, 100],
            n_cat=3,
            n_cont=5,
            out_sz=4,
            final_activation=x->FastAI.sigmoidrange(x, 2, 5)
        )
        y2 = tm2(x)
        @test all(y2.> 2) && all(y2.<5)
    end
end


