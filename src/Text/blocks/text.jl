"""
    TextBlock() <: Block

[`Block`](#) for a text paragraph containing one or more
sentences (basically, a single observation in the textual dataset). 
`data` is valid for `TextBlock` if it is of type string.

Example valid TextBlocks:

```julia
@test checkblock(TextBlock(), "Hello world!")
@test checkblock(TextBlock(), "Hello world!, How are you?")
```

"""

struct TextBlock <: Block end

checkblock(::TextBlock, ::String) = true
