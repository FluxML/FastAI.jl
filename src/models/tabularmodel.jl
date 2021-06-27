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

function TabularModel(
        layers; 
        emb_szs,
        n_cont,
        out_sz,
        ps=0,
        embed_p=0.,
        use_bn=true,
        bn_final=false,
        bn_cont=true,
        act_cls=Flux.relu,
        lin_first=true,
    final_activation=identity)

    embedslist = [Embedding(ni, nf) for (ni, nf) in emb_szs]
    n_emb = sum(size(embedlayer.weight)[1] for embedlayer in embedslist)
    #     n_emb = first(Flux.outputsize(embeds, (length(emb_szs), 1)))
    emb_drop = Dropout(embed_p)
    embeds = Chain(
        x -> collect(eachrow(x)), 
        x -> ntuple(i -> x[i], length(x)), 
        Parallel(vcat, embedslist), 
        emb_drop
    )

    bn_cont = bn_cont && n_cont>0 ? BatchNorm(n_cont) : identity

    ps = Iterators.cycle(ps)
    classifiers = []

    first_ps, ps = Iterators.peel(ps)
    push!(classifiers, linbndrop(n_emb+n_cont, first(layers); use_bn=use_bn, p=first_ps, lin_first=lin_first, act=act_cls))
    for (isize, osize, p) in zip(layers[1:(end-1)], layers[2:(end)], ps)
        layer = linbndrop(isize, osize; use_bn=use_bn, p=p, act=act_cls, lin_first=lin_first)
        push!(classifiers, layer)
    end
    push!(classifiers, linbndrop(last(layers), out_sz; use_bn=bn_final, lin_first=lin_first))
    layers = Chain(
        x -> tuple(x...),
        Parallel(vcat, embeds, Chain(x -> ndims(x)==1 ? Flux.unsqueeze(x, 2) : x, bn_cont)),
        classifiers...,
        final_activation
    )
    layers
end
