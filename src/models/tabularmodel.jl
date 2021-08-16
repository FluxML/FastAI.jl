function emb_sz_rule(n_cat)
    min(600, round(1.6 * n_cat^0.56))
end

function _one_emb_sz(cardinalitydict, catcol, sz_dict=nothing)
    sz_dict = isnothing(sz_dict) ? Dict() : sz_dict
    n_cat = cardinalitydict[catcol]
    sz = catcol in keys(sz_dict) ? sz_dict[catcol] : emb_sz_rule(n_cat)
    Int64(n_cat)+1, Int64(sz)
end

"""
    get_emb_sz(cardinalitydict, cols; sz_dict=nothing)

Returns a collection of tuples containing embedding dimensions for each column 
in `cols`. `cardinalitydict` is an indexable collection which maps to the cardinality
for each column present in `cols`. 

## Keyword arguments

- `sz_dict`: An indexable collection which may contain the required embedding 
    size for a particular column present in `cols`. If not passed, then the size is
    calculated using fastai's rule of thumb for embedding dimensions.
"""

function get_emb_sz(cardinalitydict, cols; sz_dict=nothing)
    map(catcol -> _one_emb_sz(cardinalitydict, catcol, sz_dict), cols)
end

function sigmoidrange(x, low, high)
    @. Flux.sigmoid(x) * (high - low) + low
end

function tabular_embedding_backbone(embedding_sizes, dropoutprob=0.)
    embedslist = [Flux.Embedding(ni, nf) for (ni, nf) in embedding_sizes]
    emb_drop = dropoutprob==0. ? identity : Dropout(dropoutprob)
    Chain(
        x -> tuple(eachrow(x)...), 
        Parallel(vcat, embedslist), 
        emb_drop
    )
end

function tabular_continuous_backbone(n_cont)
    BatchNorm(n_cont)
end

"""
    TabularModel(catbackbone, contbackbone, [finalclassifier]; kwargs...)
    TabularModel(catcols, `n_cont::Number, outsz::Number[; kwargs...)

Create a tabular model which takes in a tuple of categorical values 
(label or one-hot encoded) and continuous values. The default categorical backbone is
a Parallel of Embedding layers corresponding to each categorical variable, and continuous
variables are just BatchNormed. The output from these backbones is then passed through
a final classifier block.

## Keyword arguments

- `outsz`: The output size of the final classifier block. For single classification tasks, 
    this would just be the number of classes and for regression tasks, this could be the
    number of target continuous variables.
- `layers`: The sizes of the hidden layers in the classifier block.
- `ps`: Dropout probability. This could either be a single number which would be used for
        for all the classifier layers, or a collection of numbers which are cycled through
        for each layer.
- `use_bn`: Boolean variable which controls whether to use batch normalization in the classifier.
- `act_cls`: The activation function to use in the classifier layers.
- `lin_first`: Controls if the linear layer comes before or after BatchNorm and Dropout.
- `cardinalitydict`: An indexable collection which maps to the cardinality for each column present
        in `catcols`.
- `sz_dict`: An indexable collection which may contain the required embedding 
    size for a particular column present in `cols`. If not passed, then the size is
    calculated using fastai's rule of thumb for embedding dimensions.
"""

function TabularModel(
        catbackbone, 
        contbackbone;
        outsize,
        layers=[200, 100],
        kwargs...)
    TabularModel(catbackbone, contbackbone, Dense(layers[end], outsz); layers=layers, kwargs...)
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
    ps = Iterators.cycle(ps)
    classifiers = []

    first_ps, ps = Iterators.peel(ps)
    push!(classifiers, linbndrop(classifierin, first(layers); use_bn=use_bn, p=first_ps, lin_first=lin_first, act=act_cls))

    for (isize, osize, p) in zip(layers[1:(end-1)], layers[2:end], ps)
        layer = linbndrop(isize, osize; use_bn=use_bn, p=p, act=act_cls, lin_first=lin_first)
        push!(classifiers, layer)
    end
    
    Chain(
        tabularbackbone,
        classifiers...,
        finalclassifier
    )
end

function TabularModel(
        catcols,
        n_cont::Number,
        outsz::Number,
        layers=[200, 100];
        cardinalitydict,
        sz_dict=nothing)
    embedszs = get_emb_sz(cardinalitydict, catcols, sz_dict=sz_dict)
    catback = tabular_embedding_backbone(embedszs)
    contback = tabular_continuous_backbone(n_cont)

    TabularModel(catback, contback; layers=layers, outsz=outsz)
end
