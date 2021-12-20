

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
    blockscol = [
        tuplemap((b, c) -> c ? "**`$(typeof(b))`**" : "`$(typeof(b))`", bs, ch) for
        (bs, ch) in zip(blocks, changed)
    ]
    if block isa Tuple
        blockscol = [join(row, ", ") for row in blockscol]
    end
    return reshape(blockscol, :, 1)
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


function describemethod(method::SupervisedMethod)
    blocks = getblocks(method)
    input, target, x, ŷ = blocks.input, blocks.target, blocks.x

    encoding = describeencodings(
        getencodings(method),
        getblocks(method).sample,
        blocknames = ["`$(summary(input))`", "`$(summary(input))`"],
        inname = "`(input, target)`",
        outname = "`(x, y)`",
    )

    decoding = describeencodings(
        getencodings(method),
        (ŷ,),
        blocknames = ["`getblocks(method).ŷ`"],
        inname = "`ŷ`",
        outname = "`target_pred`",
        decode = true,
    )

    s = """
    #### `LearningMethod` summary

    - Task: `$(summary(input)) -> $(summary(target))`
    - Model blocks: `$(summary(x)) -> $(summary(ŷ))`

    Encoding a sample (`encode(method, context, sample)`)

    $encoding

    Decoding a model output (`decode(method, context, ŷ)`)

    $decoding
    """

    return Markdown.parse(s)
end
