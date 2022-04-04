"""
    Paragraph() <: Block

[`Block`](#) for a text paragraph containing one or more
sentences (basically, a single observation in the textual dataset). 
`data` is valid for `Paragraph` if it is of type string.

Example valid Paragraphs:

```julia
@test checkblock(Paragraph(), "Hello world!")
@test checkblock(Paragraph(), "Hello world!, How are you?")
```

"""

struct Paragraph <: Block end

FastAI.checkblock(::Paragraph, ::String) = true
