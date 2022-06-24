struct TextEncoding <: Encoding
    tfms
end

function encodedblock(p::TextEncoding, block::Paragraph)
    return block
end

function encode(p::TextEncoding, context, block::Paragraph, obs)
    return map(p.tfms, obs)
end