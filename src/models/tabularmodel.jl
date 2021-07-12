function emb_sz_rule(n_cat)
	min(600, round(1.6 * n_cat^0.56))
end

function _one_emb_sz(catdict, catcol::Symbol, sz_dict=nothing)
	sz_dict = isnothing(sz_dict) ? Dict() : sz_dict
	n_cat = length(catdict[catcol])
	sz = catcol in keys(sz_dict) ? sz_dict[catcol] : emb_sz_rule(n_cat)
	Int64(n_cat), Int64(sz)
end

function get_emb_sz(catdict, cols; sz_dict=nothing)
	[_one_emb_sz(catdict, catcol, sz_dict) for catcol in cols]
end

# function get_emb_sz(td::TableDataset, sz_dict=nothing)
# 	cols = Tables.columnaccess(td.table) ? Tables.columnnames(td.table) : Tables.columnnames(Tables.rows(td.table)[1])
# 	[_one_emb_sz(catdict, catcol, sz_dict) for catcol in cols]
# end

function sigmoidrange(x, low, high)
	@. Flux.sigmoid(x) * (high - low) + low
end

function embeddingbackbone(embedding_sizes, dropoutprob=0.)
    embedslist = [Embedding(ni => nf) for (ni, nf) in embedding_sizes]
    emb_drop = Dropout(dropoutprob)
    Chain(
        x -> tuple(eachrow(x)...), 
        Parallel(vcat, embedslist), 
        emb_drop
    )
end

function continuousbackbone(n_cont)
    n_cont > 0 ? BatchNorm(n_cont) : identity
end

function TabularModel(
        catbackbone,
        contbackbone,    
        layers=[200, 100]; 
        n_cat,
        n_cont,
        out_sz,
        ps=0,
        use_bn=true,
        bn_final=false,
        act_cls=Flux.relu,
        lin_first=true,
        final_activation=identity
    )

    tabularbackbone = Parallel(vcat, catbackbone, contbackbone)
    
    catoutsize = first(Flux.outputsize(catbackbone, (n_cat, 1)))
    ps = Iterators.cycle(ps)
    classifiers = []

    first_ps, ps = Iterators.peel(ps)
    push!(classifiers, linbndrop(catoutsize+n_cont, first(layers); use_bn=use_bn, p=first_ps, lin_first=lin_first, act=act_cls))
    
    for (isize, osize, p) in zip(layers[1:(end-1)], layers[2:(end)], ps)
        layer = linbndrop(isize, osize; use_bn=use_bn, p=p, act=act_cls, lin_first=lin_first)
        push!(classifiers, layer)
    end
    
    push!(classifiers, linbndrop(last(layers), out_sz; use_bn=bn_final, lin_first=lin_first))
    
    layers = Chain(
        tabularbackbone,
        classifiers...,
        final_activation
    )
end
