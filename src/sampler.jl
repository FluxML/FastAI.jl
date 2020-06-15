#=
sampler.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the Pytorch Sampler API to Julia

Utility methods for selecting various subsets of Datasets

The original source is here

https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#Sampler

The documentation is copied from here

https://pytorch.org/docs/stable/data.html

The main purpose of this code is to see if the team likes the method
of defining an interface and implementations in Julia
=#

using Random
using StatsBase

"""
Abstract type for all Samplers.

Samplers are iterable types, with integer elements.  Those elements
are indexes into a data source.  Where data sources are any indexable type e.g. MapDataset
    
In Julia duck typing, implementing an interface just requires 
implementing a set of required fuctions.

Required functions for Samplers are:

iterate(s<:Sampler)		    Returns either a tuple of the first index into data source
                            and initial state, or nothing if empty
iterate(s<:Sampler, state)	Returns either a tuple of the next index and next state,
                            or nothing if no items remain

Optional function:

length(s<:Sampler)          returns the number of indexes in this Sampler.  


See Julia iteration documentation for more information

    https://docs.julialang.org/en/v1/manual/interfaces/
""" 
abstract type Sampler end

"Samples elements sequentially, always in the same order."
struct SequentialSampler <: Sampler
    n::Int
end
SequentialSampler(data_source) = SequentialSampler(length(data_source))

iterate(s::SequentialSampler) = if s.n < 1 nothing else (1,2) end
iterate(s::SequentialSampler, state) = if state > length(s) nothing else (state,state+1) end
Base.length(s::SequentialSampler) = s.n

"Shared code between Random and Weighted Samplers"
abstract type IterableSampler <: Sampler end
function iterate(s::IterableSampler)
    if length(s) < 1
        return nothing
    else
        i = ids(s)
        return (i[1],(i,2))
    end
end
iterate(s::IterableSampler, state) = nothing ? state[2] > length(s) : (state[1][state[2]],state[2]+1)

"""
Samples elements randomly without replacement. Sample from a shuffled dataset.

Arguments:
        data_source: dataset to sample from
"""
struct RandomSamplerWithoutReplacement <: Sampler
    n::Int
end
RandomSamplerWithoutReplacement(data_source) = RandomSamplerWithoutReplacement(length(data_source))
ids(s::RandomSamplerWithoutReplacement) = randperm(length(s))
Base.length(s::RandomSamplerWithoutReplacement) = s.n


struct RandomSamplerWithReplacement <: Sampler
    n::Int
    num_samples::Int
end

"""
Samples elements randomly with replacement.

Arguments:
        data_source: dataset to sample from
        num_samples: number of samples to draw, default=`length(dataset)`. 
"""
function RandomSamplerWithReplacement(data_source,num_samples::Int)
    if length(data_source) < num_samples
        @error "Data source must have at least as many samples as num_samples"
    end
    if num_samples < 1
        @error "num_samples must be a positive integer"
    end
    return RandomSamplerWithReplacement(length(data_source),num_samples)
end
ids(s::RandomSamplerWithReplacement)=rand(1:length(s),s.num_samples)
Base.length(s::RandomSamplerWithReplacement) = s.num_samples

RandomSampler(data_source,num_samples=nothing,replacement::Bool =false) = if(replacement) RandomSamplerWithReplacement(data_source,num_samples) else RandomSamplerWithReplacement(data_source) end
"""
SubsetRandomSampler

Samples elements randomly from a given list of indices, without replacement.

Arguments:
        indices: an indexable sequence of indices
"""
struct SubsetRandomSampler <: Sampler
    indicies::AbstractArray{Int}
end

ids(s::SubsetRandomSampler) = shuffle(s.indicies)
Base.length(s::SubsetRandomSampler) = length(s.indicies)

"""
WeightedRandomSampler
    r'Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
"""
struct WeightedRandomSamplerWithReplacement <: Sampler
    weights::AbstractArray{Real}
    num_samples::Int
end
ids(s::WeightedRandomSamplerWithReplacement) = sample(1:s.num_samples,s.weights)
Base.length(s::WeightedRandomSamplerWithReplacement) = s.num_samples

struct WeightedRandomSamplerWithoutReplacement <: Sampler
    weights::AbstractArray{Real}
    num_samples::Int
end
ids(s::WeightedRandomSamplerWithoutReplacement) = sample(1:s.num_samples,s.weights,replace=false)
Base.length(s::WeightedRandomSamplerWithoutReplacement) = s.num_samples

"""
BatchSampler
    Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler: Base sampler.
        batch_size: Size of mini-batch.
        drop_last: If ``true``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
"""

struct BatchSampler <: Sampler
    sampler :: Sampler
    batch_size :: Int
    drop_last :: Bool
end

function batch(s::BatchSampler, offset::Int)
    batch = []
    for i in 1:s.batch_size
        if offset+i > length(s.sampler)
            return batch,offset+i
        else
            push!(batch,s.sampler[offset+i])
        end
    end
    return batch,offset+i
end

iterate(s::BatchSampler) =
    if s.drop_last
        if length(s.sampler) < s.batch_size
            return nothing
        else
            return batch(s,0)
        end
    else
        return batch(s,0)
    end

iterate(s::BatchSampler, state) =
    if s.drop_last
        if length(s.sampler) < s.batch_size + state[2]
            return nothing
        else
            return batch(s,state[2])
        end
    else
        return batch(s,0)
    end

Base.length(s::BatchSampler) = 
    if s.drop_last 
        length(s.sampler) ÷ s.batch_size 
    else 
        (length(s.sampler) + s.batch_size - 1) ÷ s.batch_size 
    end
