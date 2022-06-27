"""
    TextEncoding() <: Encoding

Encodes `Paragraph`s by applying various textual transforms.


Encodes
- `Paragraph` -> `Paragraph`

"""
struct TextEncoding <: Encoding
    tfms
end

function TextEncoding()
    base_tfms = [
        replace_all_caps,
        replace_sentence_case,
        convert_lowercase,
        remove_punctuations,
        basic_preprocessing,
        remove_extraspaces,
        # tokenize
    ]
    return TextEncoding(base_tfms)
end

function encodedblock(p::TextEncoding, block::Paragraph)
    return block
end

function encode(p::TextEncoding, context, block::Paragraph, obs)
    for tfm in values(p.tfms)
        obs = tfm(obs)
    end
    obs
end