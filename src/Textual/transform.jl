"""
    replace_all_caps(String)

Replace tokens in ALL CAPS by their lower version and add xxup before.
"""

function replace_all_caps(t)
    t = replace(t, r"([A-Z]+[^a-z\s]*)(?=(\s|$))" => s"xxup \1")
    return replace(t, r"([A-Z]*[^a-z\s]+)(?=(\s|$))" => lowercase)
end

"""
    replace_sentence_case(String)

Replace tokens in Sentence Case by their lower verions and add xxmaj before.
"""
function replace_sentence_case(t)
    t = replace(t, r"(?<!\w)([A-Z][A-Z0-9]*[a-z0-9]+)(?!\w)" => s"xxmaj \1")
    return replace(t, r"(?<!\w)([A-Z][A-Z0-9]*[a-z0-9]+)(?!\w)" => lowercase)
end

convert_lowercase(t) = string("xxbos ", lowercase(t))


## Tests


@testset "Text Transforms" begin
    str1 = "Hello WORLD CAPITAL Sentence Case"

    @test replace_all_caps(str1) == "Hello xxup world xxup capital Sentence Case"
    @test replace_sentence_case(str1) == "xxmaj hello WORLD CAPITAL xxmaj sentence xxmaj case"
    @test convert_lowercase(str1) == "xxbos hello world capital sentence case"
end
