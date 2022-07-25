function load_batchseq(data, task; context=Training(), batch_size=4, shuffle=true)
    # Create a task dataset from the data
    data = shuffle ? shuffleobs(data) : data
    td =  taskdataset(data, task, context)
    x_inp = map(i -> td[i][1], 1:length(td))
    y_out = mapreduce(i -> td[i][2], hcat, 1:length(td))

    bv_x = BatchView(x_inp, batchsize=batch_size)
    bv_y = BatchView(y_out, batchsize=batch_size)

    batches = map((xs, ys) -> (batchseq(xs, 2), ys), bv_x, bv_y)
end


function load_sample(batches; batch_index=1, sample_seq_index=1, sample_index=1)
    seq = []
    for i in batches[batch_index][sample_seq_index]
        seq = push!(seq, i[sample_index])
    end

    return (seq, batches[batch_index][2][sample_seq_index])
end
