struct TabularModel
	embeds
	emb_drop
	bn_cont
	n_emb
	n_cont
	layers
end

function TabularModel(
		layers; 
		emb_szs,
		n_cont,
		out_sz,
		ps::Union{Tuple, Vector, Number, Nothing}=nothing,
		embed_p::Float64=0.,
		y_range=nothing,
		use_bn::Bool=true,
		bn_final::Bool=false,
		bn_cont::Bool=true,
		act_cls=Flux.relu,
		lin_first::Bool=true)

	n_cont = Int64(n_cont)
	if isnothing(ps)
		ps = zeros(length(layers))
	end
	if ps isa Number
		ps = fill(ps, length(layers))
	end
	embedslist = [Embedding(ni, nf) for (ni, nf) in emb_szs]
	emb_drop = Dropout(embed_p)
	bn_cont = bn_cont ? BatchNorm(n_cont) : false
	n_emb = sum(size(embedlayer.weight)[1] for embedlayer in embedslist)
	sizes = append!(zeros(0), [n_emb+n_cont], layers, [out_sz])
	actns = append!([], [act_cls for i in 1:(length(sizes)-1)], [nothing])
	_layers = [linbndrop(Int64(sizes[i]), Int64(sizes[i+1]), use_bn=(use_bn && ((i!=(length(actns)-1)) || bn_final)), p=p, act=a, lin_first=lin_first) for (i, (p, a)) in enumerate(zip(push!(ps, 0.), actns))]
	if !isnothing(y_range)
		push!(_layers, Chain(@. x->Flux.sigmoid(x) * (y_range[2] - y_range[1]) + y_range[1]))
	end
	layers = Chain(_layers...)
	TabularModel(embedslist, emb_drop, bn_cont, n_emb, n_cont, layers)
end

function (tm::TabularModel)(x)
	x_cat, x_cont = x
	if tm.n_emb != 0
		x = [e(x_cat[i, :]) for (i, e) in enumerate(tm.embeds)]
		x = vcat(x...)
		x = tm.emb_drop(x)
	end
	if tm.n_cont != 0
		if (tm.bn_cont != false)
			x_cont = tm.bn_cont(x_cont)
		end
		x = tm.n_emb!=0 ? vcat(x, x_cont) : x_cont
	end
	tm.layers(x)
end