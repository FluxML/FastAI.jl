

function listencodeblocks(encodings, blocks)
    filledblocks = Any[blocks]
    changedblocks = Any[tuplemap(_ -> false, blocks)]
    for encoding in encodings
        push!(changedblocks, tuplemap(x -> !isnothing(x), encodedblock(encoding, blocks)))
        blocks = encodedblockfilled(encoding, blocks)
        push!(filledblocks, blocks)
    end
    return filledblocks, changedblocks
end

function listdecodeblocks(encodings, blocks)
    filledblocks = Any[blocks]
    changedblocks = Any[tuplemap(_ -> false, blocks)]
    for encoding in Iterators.reverse(encodings)
        push!(changedblocks, tuplemap(x -> !isnothing(x), decodedblock(encoding, blocks)))
        blocks = decodedblockfilled(encoding, blocks)
        push!(filledblocks, blocks)
    end
    return filledblocks, changedblocks
end

tuplemap(f, args...) = f(args...)
tuplemap(f, args::Vararg{Tuple}) = map((as...) -> tuplemap(f, as...), args...)

function blockcolumn(encodings, block; decode = false)
    blocks, changed =
        decode ? listdecodeblocks(encodings, block) : listencodeblocks(encodings, block)
    n = length(blocks)
    blockscol = [
        tuplemap((b, c) -> _blockcell(b, c, i), bs, ch) for
        (i, bs, ch) in zip(1:n, blocks, changed)
    ]
    if block isa Tuple
        blockscol = [join(row, ", ") for row in blockscol]
    end
    return reshape(blockscol, :, 1)
end

function _blockcell(block, haschanged, i)
    if haschanged
        return "**`$(summary(block))`**"
    elseif i == 1
        return "`$(summary(block))`"
    else
        return ""
    end
end

encodingscolumn(encodings) =
    reshape(["", ["`$(typeof(enc).name.name)`" for enc in encodings]...], :, 1)


function describeencodings(
    encodings,
    blocks::Tuple;
    inname = "Input",
    outname = "Output",
    blocknames = repeat([""], length(blocks)),
    decode = false,
    markdown = false,
    tf = tf_markdown,
)
    namescol = reshape([inname, ["" for _ = 2:length(encodings)]..., outname], :, 1)

    data = hcat(
        encodingscolumn(decode ? reverse(encodings) : encodings),
        namescol,
        [blockcolumn(encodings, block; decode = decode) for block in blocks]...,
    )

    s = pretty_table(
        String,
        data,
        header = [decode ? "Decoding" : "Encoding", "Name", blocknames...],
        alignment = [:r, :r, [:l for _ = 1:length(blocknames)]...],
        tf = tf,
    )
    return markdown ? Markdown.parse(s) : s
end


function describetask(task::SupervisedMethod)
    blocks = getblocks(task)
    input, target, x, ŷ = blocks.input, blocks.target, blocks.x, blocks.ŷ

    encoding = describeencodings(
        getencodings(task),
        getblocks(task).sample,
        blocknames = ["`blocks.input`", "`blocks.target`"],
        inname = "`(input, target)`",
        outname = "`(x, y)`",
    )

    s = """
    **`SupervisedMethod` summary**

    Learning task for the supervised task with input `$(summary(input))` and
    target `$(summary(target))`. Compatible with `model`s that take in
    `$(summary(x))` and output `$(summary(ŷ))`.

    Encoding a sample (`encodesample(task, context, sample)`) is done through
    the following encodings:

    $encoding
    """

    return Markdown.parse(s)
end

function describetask(task::BlockMethod)
    blocks = getblocks(task)

    encoding = describeencodings(
        getencodings(task),
        (blocks.sample,),
        blocknames = ["sample"],
        inname = "`sample`",
        outname = "`encodedsample`",
    )

    s = """
    **`BlockMethod` summary**

    Learning task with blocks

    $(join(["- $k: $(summary(v))" for (k, v) in zip(keys(blocks), values(blocks))], '\n'))

    Encoding a sample (`encodesample(task, context, sample)`) is done through
    the following encodings:

    $encoding
    """

    return Markdown.parse(s)
end
