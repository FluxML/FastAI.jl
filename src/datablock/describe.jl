

function listencodeblocks(encodings, blocks)
    filledblocks = Any[blocks]
    changedblocks = Any[tuplemap(_ -> false, blocks)]
    for encoding in encodings
        push!(changedblocks, tuplemap(x -> !isnothing(x), encodedblock(encoding, blocks)))
        blocks = encodedblock(encoding, blocks, true)
        push!(filledblocks, blocks)
    end
    return filledblocks, changedblocks
end

function listdecodeblocks(encodings, blocks)
    filledblocks = Any[blocks]
    changedblocks = Any[tuplemap(_ -> false, blocks)]
    for encoding in Iterators.reverse(encodings)
        push!(changedblocks, tuplemap(x -> !isnothing(x), decodedblock(encoding, blocks)))
        blocks = decodedblock(encoding, blocks, true)
        push!(filledblocks, blocks)
    end
    return filledblocks, changedblocks
end

tuplemap(f, args...) = f(args...)
tuplemap(f, args::Vararg{Tuple}) = map((as...) -> tuplemap(f, as...), args...)

function blockcolumn(encodings, block; decode = false)
    blocks, changed = decode ? listdecodeblocks(encodings, block) : listencodeblocks(encodings, block)
    blockscol = [tuplemap((b, c) -> c ? "**`$(typeof(b))`**" : "`$(typeof(b))`", bs, ch) for (bs, ch) in zip(blocks, changed)]
    if block isa Tuple
        blockscol = [join(row, ", ") for row in blockscol]
    end
    return reshape(blockscol, :, 1)
end

encodingscolumn(encodings) = reshape(
    ["", ["`$(typeof(enc).name.name)`" for enc in encodings]...], :, 1)


function describeencodings(
        encodings,
        blocks::Tuple;
        inname="Input",
        outname="Output",
        blocknames=repeat([""], length(blocks)),
        decode=false)
    namescol = reshape([inname, ["" for _ in 2:length(encodings)]..., outname], :, 1)

    data = hcat(
        encodingscolumn(decode ? reverse(encodings) : encodings),
        namescol,
        [blockcolumn(encodings, block; decode=decode) for block in blocks]...,
    )

    s = pretty_table(
        String,
        data,
        header=[decode ? "Decoding" : "Encoding", "Name", blocknames...],
        alignment=[:r, :r, [:l for _ in 1:length(blocknames)]...],
        tf=tf_markdown,
        )
    return Markdown.parse(s)
end


function describemethod(method::BlockMethod)
    xblock = encodedblock(method.encodings, method.blocks[1])
    s = """
    #### `LearningMethod` summary

    - Task: `$(typeof(method.blocks[1])) -> $(typeof(method.blocks[2]))`
    - Model blocks: `$(typeof(xblock)) -> $(typeof(method.outputblock))`

    Encoding a sample (`encode(method, context, sample)`)
    """
    display(Markdown.parse(s))
    display(describeencodings(
        method.encodings,
        method.blocks,
        blocknames=["`method.blocks[1]`", "`method.blocks[2]`"],
        inname="`(input, target)`", outname="`(x, y)`"))
    display(Markdown.parse("Decoding a model output (`decode(method, context, ŷ)`)"))
    display(describeencodings(
        method.encodings,
        (method.outputblock,),
        blocknames=["`method.outputblock`"],
        inname="`ŷ`", outname="`target_pred`", decode=true))


end
