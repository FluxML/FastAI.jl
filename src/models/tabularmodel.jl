function emb_sz_rule(n_cat)
    min(600, round(1.6 * n_cat^0.56))
end

function _one_emb_sz(catdict, catcol::Symbol, sz_dict=nothing)
    sz_dict = isnothing(sz_dict) ? Dict() : sz_dict
    n_cat = length(catdict[catcol])
    sz = catcol in keys(sz_dict) ? sz_dict[catcol] : emb_sz_rule(n_cat)
    Int64(n_cat)+1, Int64(sz)
end

function get_emb_sz(catdict, cols; sz_dict=nothing)
    [_one_emb_sz(catdict, catcol, sz_dict) for catcol in cols]
end

function sigmoidrange(x, low, high)
    @. Flux.sigmoid(x) * (high - low) + low
end

function embeddingbackbone(embedding_sizes, dropoutprob=0.)
    embedslist = [Flux.Embedding(ni, nf) for (ni, nf) in embedding_sizes]
    emb_drop = dropoutprob==0. ? identity : Dropout(dropoutprob)
    Chain(
        x -> tuple(eachrow(x)...), 
        Parallel(vcat, embedslist), 
        emb_drop
    )
end

function continuousbackbone(n_cont)
    n_cont > 0 ? BatchNorm(n_cont) : identity
end

function classifierbackbone(
        layers;
        ps=0,
        use_bn=true,
        bn_final=false,
        act_cls=Flux.relu,
        lin_first=true)
    ps = Iterators.cycle(ps)
    classifiers = []

    for (isize, osize, p) in zip(layers[1:(end-1)], layers[2:end], ps)
        layer = linbndrop(isize, osize; use_bn=use_bn, p=p, act=act_cls, lin_first=lin_first)
        push!(classifiers, layer)
    end
    Chain(classifiers...)
end

function TabularModel(
        catbackbone,
        contbackbone,
        classifierbackbone; 
        final_activation=identity)
    tabularbackbone = Parallel(vcat, catbackbone, contbackbone)
    Chain(
        tabularbackbone,
        classifierbackbone,
        final_activation
    )
end

function TabularModel(
        catcols,
        n_cont::Number,
        out_sz::Number,
        layers=[200, 100];
        catdict,
        sz_dict=nothing,
        ps=0.)
    embedszs = get_emb_sz(catdict, catcols, sz_dict=sz_dict)
    catback = embeddingbackbone(embedszs)
    contback = continuousbackbone(n_cont)

    classifierin = mapreduce(layer -> size(layer.weight)[1], +, catback[2].layers, init = n_cont)
    layers = append!([classifierin], layers, [out_sz])
    classback = classifierbackbone(layers, ps=ps)

    TabularModel(catback, contback, classback)
end
