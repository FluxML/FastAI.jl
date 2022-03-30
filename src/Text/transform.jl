"""
    replace_all_caps(String)

Replace tokens in ALL CAPS by their lower version and add xxup before.
"""

function replace_all_caps(t::String)
    t = replace(t, r"([A-Z]+[^a-z\s]*)(?=(\s|$))" => s"xxup \1")
    t = replace(t, r"([A-Z]*[^a-z\s]+)(?=(\s|$))" => lowercase)
end

"""
    replace_sentence_case(String)

Replace tokens in Sentence Case by their lower verions and add xxmaj before.
"""

function replace_sentence_case(t::String)
    t = replace(t, r"(?<!\w)([A-Z][A-Z0-9]*[a-z0-9]+)(?!\w)" => s"xxmaj \1")
    t = replace(t, r"(?<!\w)([A-Z][A-Z0-9]*[a-z0-9]+)(?!\w)" => lowercase)
end
