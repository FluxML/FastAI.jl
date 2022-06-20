
function showblock!(ax, ::ShowMakie, block::Label, obs)
    Makie.text!(ax, string(obs), align = (:center, :center), justification=:left,
                textsize=30, color=:gray30, markerspace=:data)
end

function showblock!(ax, ::ShowMakie, block::LabelMulti, obs)
    Makie.text!(ax, join(string.(obs), "\n"), align = (:center, :center),
                justification=:left, textsize=30, color=:gray30, markerspace=:data)
end


function showblock!(
    grid,
    ::ShowMakie,
    block::Union{<:OneHotTensor{0},<:OneHotTensorMulti{0}},
    obs,
)
    if !(sum(obs) ≈ 1)
        obs = softmax(obs)
    end
    ax = Makie.Axis(grid[1, 1], yticks = (1:length(block.classes), string.(block.classes)))
    Makie.barplot!(ax, obs, direction = :x)
    Makie.hidespines!(ax)
end
