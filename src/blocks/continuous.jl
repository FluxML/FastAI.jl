
"""
    Continuous(size) <: Block

`Block` for collections of numbers. `obs` is a valid observation
if it's length is `size` and contains `Number`s.
"""
struct Continuous <: Block
    size::Int
end

function checkblock(block::Continuous, x)
    block.size == length(x) && eltype(x) <: Number
end

mockblock(block::Continuous) = rand(block.size)

function blocklossfn(outblock::Continuous, yblock::Continuous)
    outblock.size == yblock.size || error("Sizes of $outblock and $yblock differ!")
    return Flux.Losses.mse
end


function invariant_checkblock(block::Continuous; blockvar = "block", obsvar = "obs", kwargs...)
    return invariant(
        __inv_checkblock_title(block, blockvar, obsvar),
        [
            Invariants.hastype_invariant(AbstractVector; var = obsvar),
            invariant("length(`$obsvar`) should be $(block.size)") do obs
                if !(length(obs) == block.size)
                    return """`$obsvar` should have `$(block.size)` features, instead
                    found a vector with `$(length(obs))` features.""" |> md
                end
            end,
            Invariants.hastype_invariant(
                Number,
                title = "`eltype($obsvar)` should be a subtype of number",
                inputfn = eltype,
            ),
        ];
        kwargs...
    )
end


@testset "Continuous [block]" begin
    inv = invariant_checkblock(Continuous(5))

    @test check(Bool, inv, zeros(5))
    @test !check(Bool, inv, "hi")
    @test !check(Bool, inv, ["hi"])
    @test !check(Bool, inv, [5])
end
