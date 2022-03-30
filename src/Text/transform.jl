# Contains the fucntions that deal behind the scenes with the two main tasks
# when preparing texts for modelling: tokenization and numericalization.

# Special tokens
# UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj"

"""
    spec_add_spaces(String)

Add spaces around / and #.
"""

function spec_add_spaces(t::String)
    replace(t, r"([/#\\])" => s" \1 ")
end

"""
    remove_useless_spaces(String)

Remove multiple spaces.
"""

function remove_useless_spaces(t::String)
    replace(t, r" {2,}" => s" ")
end

"""
    fix_html(String)

Remove various HTML tags and cleanup.
"""

function fix_html(t::String)
    replace(t, r"<br />" => s"\n")    
end

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
