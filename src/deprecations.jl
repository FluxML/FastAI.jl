# From renaming method -> task
Base.@deprecate methodmodel(args...; kwargs...) taskmodel(args...; kwargs...)
Base.@deprecate methoddataset(args...; kwargs...) taskdataset(args...; kwargs...)
Base.@deprecate methoddataloaders(args...; kwargs...) taskdataloaders(args...; kwargs...)
Base.@deprecate methodlossfn(args...; kwargs...) tasklossfn(args...; kwargs...)
Base.@deprecate BlockMethod(args...; kwargs...) BlockTask(args...; kwargs...)
Base.@deprecate describemethod(args...; kwargs...) describetask(args...; kwargs...)
Base.@deprecate findlearningmethods(args...; kwargs...) findlearningtasks(args...; kwargs...)
Base.@deprecate methodlearner(args...; kwargs...) tasklearner(args...; kwargs...)
Base.@deprecate savemethodmodel(args...; kwargs...) savetaskmodel(args...; kwargs...)
Base.@deprecate loadmethodmodel(args...; kwargs...) loadtaskmodel(args...; kwargs...)



"""
	findlearningtasks(blocks)

Find learning tasks compatible with block types `blocks`.

!!! warning "Deprecated"

    This function is deprecated and will be removed in a future version
    of FastAI.jl. Use `learningtasks(; blocks)` instead.


#### Examples

```julia-repl
julia> findlearningtasks((Image, Label))
[ImageClassificationSingle,]

julia> findlearningtasks((Image, Any))
[ImageClassificationSingle, ImageClassificationMulti, ImageSegmentation, ImageKeypointRegression, ...]
```
"""
findlearningtasks(blocktypes) = learningtasks(blocks=blocktypes).data.instance


@testset "Datasets [registry]" begin

    @testset "listdatasources" begin
        @test length(listdatasources()) > 1
    end

    @testset "datasetpath" begin
        @test_nowarn datasetpath("mnist_var_size_tiny")
    end

    @testset "finddatasets" begin
        @test finddatasets() |> length >= 1
        @test finddatasets(name="mnist_var_size_tiny") |> length >= 1
        @test finddatasets(blocks=Tuple{Image, Label}) |> length >= 1
        @test finddatasets(blocks=Tuple{Image, LabelMulti}) |> length >= 0
        @test finddatasets(name="mnist_var_size_tiny", blocks=Tuple{Image, Label}) |> length >= 1
        @test finddatasets(name="mnist", blocks=Tuple{Image, Label}) |> length >= 0
    end
end
