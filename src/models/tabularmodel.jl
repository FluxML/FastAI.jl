"""
    emb_sz_rule(n_cat)

Returns an embedding size corresponding to the number of classes for a 
categorical variable using the rule of thumb present in python fastai.
(see https://github.com/fastai/fastai/blob/2742fe844573d06e700f869839fb9ec5f3a9bca9/fastai/tabular/model.py#L12)
"""
emb_sz_rule(n_cat) = min(600, round(Int, 1.6 * n_cat^0.56))

"""
    get_emb_sz(cardinalities, [size_overrides])
    get_emb_sz(cardinalities; catcols, [size_overrides])

Returns a collection of tuples containing embedding dimensions corresponding to 
number of classes in categorical columns present in `cardinalities` and adjusting for NaNs. 

## Keyword arguments

- `size_overrides`: Depending on the method used, this could either be a collection of 
    Integers and `nothing` or an indexable collection with column name as key and size
    to override it with as the value. In the first case, the integer present at any index 
    will be used to override the rule of thumb for getting embedding sizes.
- `categorical_cols`: A collection of categorical column names.
"""

get_emb_sz(cardinalities::AbstractVector{<:Integer}, size_overrides=fill(nothing, length(cardinalities))) =
    map(zip(cardinalities, size_overrides)) do (cardinality, override)
        emb_dim = isnothing(override) ? emb_sz_rule(cardinality + 1) : Int64(override)
        return (cardinality + 1, emb_dim)
    end

function get_emb_sz(cardinalities; catcols, size_overrides=Dict())
    keylist = keys(size_overrides)
    overrides = map(catcols) do col
        col in keylist ? size_overrides[col] : nothing
    end
    get_emb_sz(cardinalities, overrides)
end

sigmoidrange(x, low, high) = @. Flux.sigmoid(x) * (high - low) + low

function tabular_embedding_backbone(embedding_sizes, dropoutprob=0.)
    embedslist = [Flux.Embedding(ni, nf) for (ni, nf) in embedding_sizes]
    emb_drop = dropoutprob==0. ? identity : Dropout(dropoutprob)
    Chain(
        x -> tuple(eachrow(x)...), 
        Parallel(vcat, embedslist), 
        emb_drop
    )
end

tabular_continuous_backbone(n_cont) = BatchNorm(n_cont)

"""
    TabularModel(catbackbone, contbackbone, [finalclassifier]; kwargs...)
    TabularModel(n_cont, outsize [; kwargs...])

Create a tabular model which takes in a tuple of categorical values 
(label or one-hot encoded) and continuous values. The default categorical backbone is
a Parallel of Embedding layers corresponding to each categorical variable, and continuous
variables are just BatchNormed. The output from these backbones is then passed through
a final classifier block.

## Keyword arguments

- `outsize`: The output size of the final classifier block. For single classification tasks, 
    this would just be the number of classes and for regression tasks, this could be the
    number of target continuous variables.
- `layersizes`: The sizes of the hidden layers in the classifier block.
- `dropout_rates`: Dropout probability. This could either be a single number which would be 
    used for for all the classifier layers, or a collection of numbers which are cycled through
    for each layer.
- `batchnorm`: Boolean variable which controls whether to use batch normalization in the classifier.
- `activation`: The activation function to use in the classifier layers.
- `linear_first`: Controls if the linear layer comes before or after BatchNorm and Dropout.
- `cardinalities`: A collection of sizes (number of classes) for each categorical column.
- `size_overrides`: An optional argument which corresponds to a collection containing 
    embedding sizes to override the value returned by the "rule of thumb" for a particular index 
    corresponding to `cardinalities`, or `nothing`.
"""

function TabularModel(
        catbackbone, 
        contbackbone;
        outsize,
        layersizes=(200, 100),
        kwargs...)
    TabularModel(catbackbone, contbackbone, Dense(layersizes[end], outsize); layersizes=layersizes, kwargs...)
end

function TabularModel(
        catbackbone,
        contbackbone,
        finalclassifier;
        layersizes=[200, 100],
        dropout_rates=0.,
        batchnorm=true,
        activation=Flux.relu,
        linear_first=true)
    
    tabularbackbone = Parallel(vcat, catbackbone, contbackbone)

    classifierin = mapreduce(layer -> size(layer.weight)[1], +, catbackbone[2].layers;
                             init = contbackbone.chs)
    dropout_rates = Iterators.cycle(dropout_rates)
    classifiers = []

    first_ps, dropout_rates = Iterators.peel(dropout_rates)
    push!(classifiers, linbndrop(classifierin, first(layersizes);
                                 use_bn=batchnorm, p=first_ps, lin_first=linear_first, act=activation))

    for (isize, osize, p) in zip(layersizes[1:(end-1)], layersizes[2:end], dropout_rates)
        layer = linbndrop(isize, osize; use_bn=batchnorm, p=p, act=activation, lin_first=linear_first)
        push!(classifiers, layer)
    end
    
    Chain(
        tabularbackbone,
        classifiers...,
        finalclassifier
    )
end

function TabularModel(
        n_cont::Number,
        outsize::Number,
        layersizes=(200, 100);
        cardinalities,
        size_overrides=fill(nothing, length(cardinalities)))
    embedszs = get_emb_sz(cardinalities, size_overrides)
    catback = tabular_embedding_backbone(embedszs)
    contback = tabular_continuous_backbone(n_cont)

    TabularModel(catback, contback; layersizes=layersizes, outsize=outsize)
end
