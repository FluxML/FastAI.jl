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

    @testset ExtendedTestSet "classifierbackbone" begin
        classback = classifierbackbone([10, 200, 100, 2])
        x = rand(10, 2)
        @test size(classback(x)) == (2, 2)
    end

    @testset ExtendedTestSet "TabularModel" begin
        n = 5
        embed_szs = [(5, 10), (100, 30), (2, 30)]
        
        embeds = embeddingbackbone(embed_szs, 0.)
        contback = continuousbackbone(n)
        classback = classifierbackbone([75, 200, 100, 4])

        tm = TabularModel(embeds, contback, classback, final_activation = x->FastAI.sigmoidrange(x, 2, 5))

        x = ([rand(1:n) for (n, _) in embed_szs], rand(5, 1))
        y1 = tm(x)
        @test size(y1) == (4, 1)
        @test all(y1.> 2) && all(y1.<5)

        catcols = [:a, :b, :c]
        catdict = Dict(:a => rand(4), :b => rand(99), :c => rand(1))
        tm2 = TabularModel(catcols, n, 4, [200, 100], catdict = catdict, sz_dict = Dict(:a=>10, :b=>30, :c=>30))
        @test size(tm2(x)) == (4, 1)
    end
end


