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

"""
    drop_mask(x, p)

Drop mask generator

This function generates dropout mask for given 'x' with `p` probability
    or
It can be used to generate the mask by giving the shape of the desired mask and probaility
"""
function drop_mask(x, p)
    y = similar(x, size(x))
    Flux.rand!(y)
    y .= Flux._dropout_kernel.(y, p, 1 - p)
    return y
end

drop_mask(shape::Tuple, p; type = Float32) = (mask = rand(type, shape...);mask .= _dropout_kernel.(mask, p, 1 - p))

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
mutable struct WeightDroppedLSTMCell{A, V, S, M}
    Wi::A
    Wh::A
    b::V
    h::S
    c::S
    p::Float64
    maskWi::M
    maskWh::M
    active::Bool
end

function WeightDroppedLSTMCell(in::Integer, out::Integer, p::Float64=0.0;
    init = Flux.glorot_uniform)
    @assert 0 ≤ p ≤ 1
    cell = WeightDroppedLSTMCell(
        init(out*4, in),
        init(out*4, out),
        init(out*4),
        reshape(zeros(Float32, out), out, 1),
        reshape(zeros(Float32, out), out, 1),
        p,
        drop_mask((out*4, in), p),
        drop_mask((out*4, out), p),
        true
    )
    cell.b[gate(out, 2)] .= 1
    return cell
end

function (m::WeightDroppedLSTMCell)((h, c), x)
    b, o = m.b, size(h, 1)
    Wi = m.active ? m.Wi .* m.maskWi : m.Wi
    Wh = m.active ? m.Wh .* m.maskWh : m.Wh
    g = Wi*x .+ Wh*h .+ b
    input = σ.(gate(g, o, 1))
    forget = σ.(gate(g, o, 2))
    cell = tanh.(gate(g, o, 3))
    output = σ.(gate(g, o, 4))
    c = forget .* c .+ input .* cell
    h′ = output .* tanh.(c)
    return (h′, c), h′
end

Flux.@functor WeightDroppedLSTMCell

Flux.trainable(m::WeightDroppedLSTMCell) = (m.Wi, m.Wh, m.b, m.h, m.c)

testmode!(m::WeightDroppedLSTMCell, mode=true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

"""
    WeightDroppedLSTM(in::Integer, out::Integer, p::Float64=0.0)

WeightDroppedLSTM layer

This makes the WeightDroppedLSTMCell stateful, by wrapping the layer in a Recur field.

Defining an instance:

julia> wd = WeightDroppedLSTM(4, 5, 0.3);
"""
function WeightDroppedLSTM(a...; kw...)
    cell = WeightDroppedLSTMCell(a...;kw...)
    hidden = (cell.h, cell.c)
    return Flux.Recur(cell, hidden)
end

"""
    reset!(m)

Resets the h, c parameters of the LSTM Cell.

For more refer [`Flux.reset`](@ref https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.reset!)
"""
function reset!(m)
    try		# to accomodate the definition in previously trained Language Model
        (m.state = (m.cell.h, m.cell.c))
    catch
    	Flux.reset!(m)
    end
    Flux.reset!(m)
end


"""
    reset_masks!(layer)

This is an important funciton since it used to reset the masks
which are saved in WeightDroppedLSTMCell after every pass.

julia> wd = WeightDroppedLSTM()

julia> reset_masks!(wd)
"""
function reset_masks!(wd::T) where T <: Flux.Recur{<:WeightDroppedLSTMCell}
    wd.cell.maskWi = drop_mask(wd.cell.Wi, wd.cell.p)
    wd.cell.maskWh = drop_mask(wd.cell.Wh, wd.cell.p)
    return
end
####################################################################

################## ASGD Weight-Dropped LSTM Layer ##################
"""
    AWD_LSTM(in::Integer, out::Integer, p::Float64=0.0)

Average SGD Weight-Dropped LSTM

This custom layer is used for training the Language model,
instead of standard LSTM layer.
This layer carries two addtional functionality:
    Weight-dropping (DropConnect)
    Averaging of weights

AWD_LSTM is basically a wrapper aroung WeightDroppedLSTM layer,
it has three fields:
    layer : WeightDroppedLSTM layer
    T     : Trigger iteration, to trigger averaging
    accum : After triggring the accumlation of weights is saved here

cite this paper to know more:
https://arxiv.org/pdf/1708.02182.pdf
"""
mutable struct AWD_LSTM
    layer::Flux.Recur
    T::Integer
    accum
end

AWD_LSTM(in::Integer, out::Integer, p::Float64=0.0; kw...) = AWD_LSTM(WeightDroppedLSTM(in, out, p; kw...), -1, [])

Flux.@functor AWD_LSTM

Flux.trainable(m::AWD_LSTM) = (m.layer,)

(m::AWD_LSTM)(in) = m.layer(in)

# To set the trigger point for the AWD_LSTM layer
set_trigger!(t, m) = nothing
set_trigger!(trigger_point::Integer, m::AWD_LSTM) = m.T = trigger_point;

# resets mask of the WeightDroppedLSTM layer contained in AWD_LSTM
reset_masks!(awd::AWD_LSTM) = reset_masks!(awd.layer)

"""
    asgd_step!(i::Integer, layer::AWD_LSTM)

Averaged Stochastic Gradient Descent Step

This funciton performs the Averaging step to the given AWD_LSTM layer,
if the trigger point or trigger iteration is reached.
Arguments:
    i       : current iteration of the training loop
    layer   : AWD_LSTM layer
"""
asgd_step!(i, l) = nothing

function asgd_step!(iter::Integer, layer::AWD_LSTM)
    if (iter >= layer.T) & (layer.T > 0)
        p = get_trainable_params([layer])
        avg_fact = 1/max(iter - layer.T + 1, 1)
        if avg_fact != 1
            layer.accum = layer.accum .+ p
            for (ps, accum) in zip(p, layer.accum)
                ps .= avg_fact*accum
            end
        else
            layer.accum = deepcopy(p)   # Accumulator for ASGD
        end
    end
    return
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

mutable struct VarDrop{F}
    p::F
    mask
    reset::Bool
    active::Bool
end

VarDrop(p::Float64=0.0) = VarDrop(p, Array{Float32, 2}(UndefInitializer(), 0, 0), true, true)

function (vd::VarDrop)(x)
    vd.active || return x
    if vd.reset
        vd.mask = drop_mask(x, vd.p)
        vd.reset = false
    end
    return (x .* vd.mask)
end

testmode!(m::VarDrop, mode=true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

# method for reseting mask of VarDrop
reset_masks!(vd::VarDrop) = (vd.reset = true)

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
mutable struct DroppedEmbeddings{A, F}
    emb::A
    p::F
    mask
    active::Bool
end

function DroppedEmbeddings(in::Integer, embed_size::Integer, p::Float64=0.0;
    init = Flux.glorot_uniform)
        de = DroppedEmbeddings{AbstractArray, typeof(p)}(
            init(in, embed_size),
            p,
            drop_mask((in,), p),
            true
        )
    return de
end

function (de::DroppedEmbeddings)(x::AbstractArray, tying::Bool=false)
    dropped = de.active ? de.emb .* de.mask : de.emb
    return tying ? dropped * x : transpose(dropped[x, :])
end

Flux.@functor DroppedEmbeddings

Flux.trainable(m::DroppedEmbeddings) = (m.emb,)

testmode!(m::DroppedEmbeddings, mode=true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

function reset_masks!(de::DroppedEmbeddings)
    de.mask = drop_mask(de.mask, de.p)
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
mutable struct PooledDense{F, S, T}
    W::S
    b::T
    σ::F
end

PooledDense(W, b) = PooledDense(W, b, identity)

function PooledDense(hidden_sz::Integer, out::Integer, σ = identity;
             initW = Flux.glorot_uniform, initb = (dims...) -> zeros(Float32, dims...))
return PooledDense(initW(out, hidden_sz*3), initb(out), σ)
end

Flux.@functor PooledDense

function (a::PooledDense)(x)
    W, b, σ = a.W, a.b, a.σ
    x = cat(x..., dims=3)
    maxpool = maximum(x, dims=3)[:, :, 1]
    meanpool = (sum(x, dims=3)/size(x, 3))[:, :, 1]
    hc = cat(x[:, :, 1], maxpool, meanpool, dims=1)
    σ.(W*hc .+ b)
end

####################################################################

"""
get_trainable_params(layers)

This funciton works same as `params` function except for `AWD_LSTM` layer.
While getting `Params` of the `AWD_LSTM` it does not include the `h` and `c` `params` of `AWD_LSTM`.
This is useful while calculating gradients because calculating gradients for `h` and `c` fields
in `AWD_LSTM` is unnecessary here.

# Example:

julia> layers = Chain(DroppedEmbeddings(4,5,0.2),
                    AWD_LSTM(5, 3),
                    Dense(3, 2),
                    softmax
                );
julia> p1 = params(layers);
julia> p2 = get_trainable_params(layers);

julia> length(p1)
8

julia> length(p2)
6

`Params` from all the other layers are included in p2 except for `h` and `c`
"""
function get_trainable_params(layers)
    p = []
    function get_awd_params(awd::AWD_LSTM)
        return [awd.layer.cell.Wi,
        awd.layer.cell.Wh,
        awd.layer.cell.b]
    end
    for layer in layers
        layer isa Array || (layer = [layer])
        for l in layer
            l isa AWD_LSTM && (append!(p, get_awd_params(l)); continue)
            push!(p, l)
        end
    end
    return Flux.params(p...)
end
