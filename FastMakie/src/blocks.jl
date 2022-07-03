
function showblock!(ax, ::ShowMakie, block::Label, obs)
    Makie.text!(ax, string(obs), align = (:center, :center), justification = :left,
                textsize = 30, color = :gray30, markerspace = :data)
end

function showblock!(ax, ::ShowMakie, block::LabelMulti, obs)
    Makie.text!(ax, join(string.(obs), "\n"), align = (:center, :center),
                justification = :left, textsize = 30, color = :gray30, markerspace = :data)
end

function showblock!(ax,
                    ::ShowMakie,
                    block::OneHotTensor{0},
                    obs)
    # if raw model output, softmax so they add to 1
    obs = sum(obs) == 1 ? obs : softmax(obs)
    Makie.barplot!(ax, obs, direction = :x)
    Makie.hidespines!(ax)
end

function showblock!(ax,
                    ::ShowMakie,
                    block::OneHotTensorMulti{0},
                    obs)
    # if raw model output, scale to between 0 and 1
    ax.subtitle = string(nameof(typeof(block)))
    l, h = extrema(obs)
    if l < 0 || h > 1
        obs = sigmoid.(obs)
    end
    Makie.xlims!(ax, (0, 1))
    obs = sum(obs) == 1 ? obs : softmax(obs)
    Makie.barplot!(ax, obs, direction = :x)
    Makie.hidespines!(ax)
end

function axiskwargs(block::Union{<:OneHotTensor{0}, <:OneHotTensorMulti{0}})
    (;
     yticks = (1:length(block.classes), string.(block.classes)),
     clean = false,
     xlabel = "Confidence",
     ylabel = "Label",
     dataaspect = false)
end

@testset "ShowMakie blocks" begin
    backend = ShowMakie()
    @testset "Label" begin @test_nowarn showblock(backend, Label(1:10), 1) end
    @testset "LabelMulti" begin @test_nowarn showblock(backend, LabelMulti(1:10), [1, 2]) end
    @testset "OneHotTensor" begin
        block = OneHotTensor{0, Float32}(1:10)
        obs = FastAI.mockblock(block)
        @test_nowarn showblock(backend, block, obs)
    end
    @testset "OneHotTensorMulti" begin
        block = OneHotTensorMulti{0, Float32}(1:10)
        obs = FastAI.mockblock(block)
        @test_nowarn showblock(backend, block, obs)
    end
end
