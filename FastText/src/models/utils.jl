"""
Helping functions
"""

# Converts vector of words to vector of indices
function indices(wordVect::Vector, vocab::Vector, unk::String="_unk_")
    function index(x, unk)
        idx = something(findfirst(isequal(x), vocab), 0)
        idx > 0 || return findfirst(isequal(unk), vocab)
        return idx
    end
    return broadcast(x -> index(x, unk), wordVect)
end

# Padding multiple sequences w r t the max size sequence
function pre_pad_sequences(sequences::Vector, pad::String="_pad_")
    max_len = maximum([length(x) for x in sequences])
    return [[fill(pad, max_len-length(sequence)); sequence] for sequence in sequences]
end

function post_pad_sequences(sequences::Vector, pad::String="_pad_")
    max_len = maximum([length(x) for x in sequences])
    return [[sequence; fill(pad, max_len-length(sequence))] for sequence in sequences]
end

# To initialize funciton for model LSTM weights
init_weights(extreme::AbstractFloat, dims...) = randn(Float32, dims...) .* sqrt(Float32(extreme))

# Generator, whenever it should be called two times since it gives X in first and y in second call
function generator(c::Channel, corpus; batchsize::Integer=64, bptt::Integer=70)
    X_total = post_pad_sequences(Flux.chunk(corpus, batchsize))
    n_batches = Int(floor(length(X_total[1])/bptt))
    put!(c, n_batches)
    for i=1:n_batches
        start = bptt*(i-1) + 1
        batch = [Flux.batch(X_total[k][j] for k=1:batchsize) for j=start:start+bptt]
        put!(c, batch[1:end-1])
        put!(c, batch[2:end])
    end
end

"""
    get_buckets(c::Corpus, bucketsize::Integer; order::Bool=true)

Simple Sequence-Bucketing

This function will return the groups of `Document`s with close sequence lengths from the given `Corpus`.
Use this function with `data_loader` function to get a `Channel`. Also, the order of sequences lengths
can be set using keyword argument `order`, which is, `true` for ascending order (default) and `false` for
descending order of lengths of sequences.

# Example:

julia> buckets = get_buckets(corpus, 32);

"""
function get_buckets(c::Corpus, labels::Vector, bucketsize::Integer; order::Bool=true)
    lengths = length.(tokens.(documents(c)))
    sorted_lens = order ? sortperm(lengths) : reverse(sortperm(lengths))
    c, labels = c[sorted_lens], labels[sorted_lens]
    buckets = []
    for i=1:bucketsize:length(c)
        (length(c)-i) < (bucketsize-1) && (push!(buckets, zip(c[i:end], labels[i:end]));continue)
        push!(buckets, zip(c[i:i+bucketsize-1], labels[i:i+bucketsize-1]))
    end
    return buckets
end

"""
    data_loader(buckets::AbstractArray, classes::Vector; pad_func::Function=pre_pad_sequences)
    data_loader(dataset::Corpus, labels::Vector, batchsize::Integer; pad_func::Function=pre_pad_sequences)

This funciton can be use to make `Channel` to load data for the training of `TextClassifier`.
Do preprocessing of the text before using this function. And get all the examples into a `Corpus` type
and all the corresponding labels into a `Vector`.

NOTE: Remember that the first call to the `Channel` will output an integer which specifies the number of batches the `Channel` can give.

It can be used in two ways:
 - With sequence bucketing: use `get_buckets` function first on the prepared dataset to get buckets
   and pass those buckets to this function. Now, make a Vector of all possible classes and pass this to the `data_loader`
   and keep this classes variable safe since the output units of the classifier will be corresponding to this `Vector`.
   If already a list of all the classes is available that can also be passed to this function.

   Remember, the output buckets from the `get_buckets` the length of the last bucket may not be equal to the batchsize specified,
   since the number of examples couldn't be division by batchsize specified. That can be passed to data_loader
   but it might cause problems later while training, so that batch can be removed if it is not a big loss of training data.
   Or pass number of examples that can be divided into batches of specified batchsize.

julia> classes = unique(labels)     # labels contains the labels for the examples
julia> buckets = get_buckets(data, labels, 32);

# To check whether the buckets are all of same size:
julia> print(length(buckets[end]))

julia> loader = data_loader(buckets, classes)

 - Without sequence bucketing: pass the `Corpus` and labels direclty to this funciton and specify batchsize.
   This will return a Channel, which will give a batch at every call with labels.

julia> loader = data_loader(data, labels, 32)

The padding can be controlled by setting the `pad_func` keyword argument to
either `pre_pad_sequences` or `post_pad_sequence`, former is for pre padding the sequence
and that latter is for post padding the sequences.
"""
function data_loader(buckets::AbstractArray, classes::Vector; pad_func::Function=pre_pad_sequences)
    shuffle!(buckets)
    Channel(csize=1) do docs
        n_batches = length(buckets)
        put!(docs, n_batches)
        for b in buckets
            shuffle!(b)
            X, Y = [], []
            for (cur_text, label) in b
                toks = tokenize(cur_text)
                push!(X, toks)
                y = Flux.onehotbatch([label], classes)
                push!(Y, y)
            end#for
            X = pad_func(X, "_pad_")
            batchsize = length(X)
            put!(docs, [Flux.batch(X[k][j] for k=1:batchsize) for j=1:length(X[1])])
            put!(docs, cat(Y..., dims=2))
        end #for
    end #channel
end

function data_loader(dataset::Corpus, labels::Vector, batchsize::Integer; pad_func::Function=pre_pad_sequences)
    classes = unique(labels)
    n_batches = Int(floor(length(dataset)/batchsize))
    Channel(csize=1) do loader
        put!(loader, n_batches)
        for i=1:n_batches
            X = tokens.(dataset[(i-1)*batchsize+1:i*batchsize])
            Y = Flux.onehotbatch(labels[(i-1)*batchsize+1:i*batchsize], classes)
            X = pad_func(X, "_pad_")
            put!(loader, [Flux.batch(X[k][j] for k=1:batchsize) for j=1:length(X[1])])
            put!(loader, Y)
        end
    end
end
