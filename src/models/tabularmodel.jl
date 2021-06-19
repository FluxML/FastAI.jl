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
        n_cont::Int64,
        out_sz,
        ps::Union{Tuple, Vector, Number}=0,
        embed_p::Float64=0.,
        y_range=nothing,
        use_bn::Bool=true,
        bn_final::Bool=false,
        bn_cont::Bool=true,
        act_cls=Flux.relu,
        lin_first::Bool=true)

    embedslist = [Embedding(ni, nf) for (ni, nf) in emb_szs]
    emb_drop = Dropout(embed_p)
    embeds = Chain(x -> ntuple(i -> x[i, :], length(emb_szs)), Parallel(vcat, embedslist...), emb_drop)

    bn_cont = bn_cont ? BatchNorm(n_cont) : identity

    n_emb = sum(size(embedlayer.weight)[1] for embedlayer in embedslist)

    ps = Iterators.cycle(ps)
    classifiers = []

    first_ps, ps = Iterators.peel(ps)
    push!(classifiers, linbndrop(n_emb+n_cont, first(layers); use_bn=use_bn, p=first_ps, lin_first=lin_first, act=act_cls))
    for (isize, osize, p) in zip(layers[1:(end-1)], layers[2:(end)], ps)
        layer = linbndrop(isize, osize; use_bn=use_bn, p=p, act=act_cls, lin_first=lin_first)
        push!(classifiers, layer)
    end
    push!(classifiers, linbndrop(last(layers), out_sz; use_bn=bn_final, lin_first=lin_first))

    layers = isnothing(y_range) ? Chain(Parallel(vcat, embeds, bn_cont), classifiers...) : Chain(Parallel(vcat, embeds, bn_cont), classifiers..., @. x->Flux.sigmoid(x) * (y_range[2] - y_range[1]) + y_range[1])
    layers
end
