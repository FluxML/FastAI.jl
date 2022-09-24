function load_batchseq(data, task; context = Training(), batch_size = 4, shuffle = true)
    # Create a task dataset from the data
    data = shuffle ? shuffleobs(data) : data
    td = taskdataset(data, task, context)
    x_inp = mapobs(i -> td[i][1], 1:numobs(td))
    y_out = mapreduce(i -> td[i][2], hcat, 1:length(td))

    bv_x = BatchView(x_inp, batchsize = batch_size)
    bv_y = BatchView(y_out, batchsize = batch_size)

    return map((xs, ys) -> (batchseq(xs, 2), ys), bv_x, bv_y)
end

function load_genseq(data, task; context = Training(), batch_size = 4, shuffle = true)
    # Create a task dataset from the data
    data = shuffle ? shuffleobs(data) : data
    td = taskdataset(data, task, context)
    x_inp = mapobs(i -> td[i][1][1:(end - 1)], 1:numobs(td))
    y_out = mapobs(i -> td[i][2][2:end], 1:numobs(td))

    bv_x = BatchView(x_inp, batchsize = batch_size)
    # bv_y = BatchView(y_out, batchsize = batch_size)
    bv_y = BatchView(y_out, batchsize = batch_size)

    pad_onehot = Flux.onehot(2, 1:length(task.encodings[3].vocab))

    return map((xs, ys) -> (batchseq(xs, 2), batchseq(ys, pad_onehot)), bv_x, bv_y)
end

function encode(::OneHot, _, block::NumberVector, obs)
    return map(i -> Flux.onehotbatch(obs[i], 1:4424), 1:length(obs))
end