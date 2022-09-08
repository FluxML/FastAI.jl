"""
ULMFiT - Custom layers

This file contains the custom layers defined for this model:
    AWD_LSTM
    WeightDroppedLSTM
    VarDrop
    PooledDense
"""

import Flux: gate, testmode!, _dropout_kernel

reset_masks!(entity) = nothing
reset_probability!(entity) = nothing


#################### Weight-Dropped LSTM Cell ######################
"""
Weight-Dropped LSTM Cell

This is an LSTM layer with dropped weights functionality, that is, DropConnect technique

cite this paper to know about DropConnec:
http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf

Moreover this also follows the Vartional DropOut citeria, that is,
the drop mask is remains same for a whole training pass.
This is done by saving the masks in 'maskWi' and 'maskWh' fields
"""
mutable struct WeightDroppedLSTMCell{A,V,S,T}
    Wi::A
    Wh::A
    b::V
    h::S
    c::S
    p::Float32
    active::Union{Bool,Nothing}
    state0::T
end

function WeightDroppedLSTMCell(in::Integer, out::Integer, p::Float32 = 0.0f0;
    init = Flux.glorot_uniform, initb = Flux.zeros32, init_state = Flux.zeros32
)

    @assert 0 ≤ p ≤ 1
    cell = WeightDroppedLSTMCell(
        init(out * 4, in),
        init(out * 4, out),
        initb(out * 4),
        reshape(zeros(Float32, out), out, 1),
        reshape(zeros(Float32, out), out, 1),
        p,
        nothing,
        ((init_state(out, 1), init_state(out, 1)))
    )
    cell.b[gate(out, 2)] .= 1
    return cell
end

function (m::WeightDroppedLSTMCell)((h, c, maskWi, maskWh), x)
    b, o = m.b, size(h, 1)
    Wi = Flux._isactive(m) ? m.Wi .* maskWi : m.Wi
    Wh = Flux._isactive(m) ? m.Wh .* maskWh : m.Wh

    g = Wi * x .+ Wh * h .+ b
    input = σ.(gate(g, o, 1))
    forget = σ.(gate(g, o, 2))
    cell = tanh.(gate(g, o, 3))
    output = σ.(gate(g, o, 4))
    c = forget .* c .+ input .* cell
    h′ = output .* tanh.(c)

    return (h′, c, maskWi, maskWh), h′
end

Flux.@functor WeightDroppedLSTMCell

Flux.trainable(m::WeightDroppedLSTMCell) = (Wi = m.Wi, Wh = m.Wh, b = m.b)

testmode!(m::WeightDroppedLSTMCell, mode = true) = (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

"""
    WeightDroppedLSTM(in::Integer, out::Integer, p::Float64=0.0)

WeightDroppedLSTM layer

This makes the WeightDroppedLSTMCell stateful, by wrapping the layer in a Recur field.

Defining an instance:

julia> wd = WeightDroppedLSTM(4, 5, 0.3);
"""
function WeightDroppedLSTM(a...; kw...)
    cell = WeightDroppedLSTMCell(a...; kw...)
    maskWi = Flux.dropout_mask(Flux.rng_from_array(cell.Wi), cell.Wi, cell.p)
    maskWh = Flux.dropout_mask(Flux.rng_from_array(cell.Wh), cell.Wh, cell.p)
    hidden = (cell.state0..., maskWi, maskWh)
    return Flux.Recur(cell, hidden)
end

function Flux.reset!(layer::Flux.Recur{<:WeightDroppedLSTMCell})
    maskWi = Flux.dropout_mask(Flux.rng_from_array(), layer.cell.Wi, layer.cell.p)
    maskWh = Flux.dropout_mask(Flux.rng_from_array(), layer.cell.Wh, layer.cell.p)
    layer.state = (layer.cell.state0..., maskWi, maskWh)
    return nothing
end
####################################################################

########################## Varitional DropOut ######################
"""
    VarDrop(p::Float64=0.0)

Variational Dropout layer

Unlike standard dropout layer, which applies new dropout mask to every Array ot Matrix passed to it,
this layer saves the mask applied till it is explicitly set to reset mode using 'reset_masks!'.

Usage:
julia> vd = VarDrop()

To reset mask:
julia> reset_masks!(vd)
"""

mutable struct VarDropCell{F}
    p::F
    active::Union{Bool,Nothing} # matches other norm layers
end
Flux.@functor VarDropCell

VarDropCell(p::Real = 0.0) = VarDropCell(p, nothing)

testmode!(m::VarDropCell, mode = true) =
    (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)


function (vd::VarDropCell)((has_mask, mask), x)
    if Flux._isactive(vd)
        mask = has_mask ? mask : Flux.dropout_mask(Flux.rng_from_array(x), x, vd.p)
        return (true, mask), x .* mask
    elseif !has_mask
        return (has_mask, mask), x
    else
        error("Mask set but layer is in test mode. Call `reset!` to clear the mask.")
    end
end

# The single-element array keeps Recur happy.
# Limitation: typeof(p) must == typeof(<inputs>)
VarDrop(p::Real) = Flux.Recur(VarDropCell(p), (false, ones(typeof(p), 1, 1)))

function Flux.reset!(m::Flux.Recur{<:VarDropCell})
    m.state = (false, ones(typeof(m.cell.p), 1, 1))
    return nothing
end


######################################################################

################# Varitional Dropped Embeddings ######################
"""
    DroppedEmbeddings(in::Integer, embed_size::Integer, p::Float64=0.0)

Embeddings with varitional dropout

This struct defines an embedding layer with Varitional Embedding dropout functionality.
Instead of randomly dropping values of embedding matrix,
this layer drops all values of a specific token, in other words,
that token is dropped from the embedding matrix for that particular pass.

Since this follows Variational DropOut criteria, it also saves the drop mask,
which should be reset to new mask explicilty using `reset_masks!` function

# Usage:
It takes input size, embedding size and dropout probability

julia> de = DroppedEmbeddings(1000, 20, 0.4)

To reset mask:

julia> reset_masks!(de)
"""
mutable struct DroppedEmbeddings{A,F,M}
    emb::A
    p::F
    mask::M
    active::Union{Bool,Nothing}
end

function DroppedEmbeddings(in::Integer, embed_size::Integer, p::Float32 = 0.0f0;
    init = Flux.glorot_uniform)
    de = DroppedEmbeddings(
        init(in, embed_size),
        p,
        Flux.dropout(Flux.rng_from_array(), rand(Float32, in), p),
        nothing
    )
    return de
end

function (de::DroppedEmbeddings)(x::AbstractArray, tying::Bool = false)
    dropped = Flux._isactive(de) ? de.emb .* de.mask : de.emb
    return tying ? dropped * x : NNlib.gather(transpose(dropped), x)
end

Flux.@functor DroppedEmbeddings

Flux.trainable(m::DroppedEmbeddings) = (; emb = m.emb)

testmode!(m::DroppedEmbeddings, mode = true) =
    (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

function reset_masks!(de::DroppedEmbeddings)
    de.mask = Flux.dropout_mask(Flux.rng_from_array(), de.mask, de.p)
    return
end
####################################################################

################# Concat Pooling Dense layer #######################
"""
    PooledDense(hidden_sz::Integer, out::Integer, σ = identity)

Concat-Pooled Dense layer

This is basically a modified version of the `Dense` layer.
It takes the `Vector` of outputs of RNN at all time-steps,
then it calculates the mean and max pools for those outputs and
concatenates output RNN at the last time-step with these max and mean pools.
Then this conatenated `Vector` is multiplied with weights and added with bias
and passes through specified activation function.

Usage:
The first argument `hidden_sz` takes length of the ouput of the preceding RNN layer.
Other two arguments are output size and activation function

# Example

julia> pd = PooledDense(40, 20)    # if the output size of the RNN layer is 40 in this case
"""
struct PooledDense{F,S,T}
    W::S
    b::T
    σ::F
end

PooledDense(W, b) = PooledDense(W, b, identity)

function PooledDense(
    hidden_sz::Integer,
    out::Integer,
    σ = identity;
    initW = Flux.glorot_uniform,
    initb = Flux.zeros32
)
    return PooledDense(initW(out, hidden_sz * 3), initb(out), σ)
end

Flux.@functor PooledDense

function (a::PooledDense)(x)
    W, b, σ = a.W, a.b, a.σ
    x = cat(x..., dims = 3)
    maxpool = maximum(x, dims = 3)[:, :, 1]
    meanpool = (sum(x, dims = 3) / size(x, 3))[:, :, 1]
    hc = cat(x[:, :, 1], maxpool, meanpool, dims = 1)
    σ.(W * hc .+ b)
end