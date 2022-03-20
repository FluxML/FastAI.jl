
"""
    TextDatasetRecipe(tablefile; catcols, contcols, kwargs...])

Recipe for loading a `TextDataset`. `tablefile` is the path of a file that can
be read as a table. `catcols` and `contcols` indicate the categorical and
continuous columns of the text table. If they are not given, they are detected
automatically.
"""