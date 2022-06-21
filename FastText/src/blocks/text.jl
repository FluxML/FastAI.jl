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

You can create a random observation using [`mockblock`](#):

{cell=main}
```julia
using FastAI
FastAI.mockblock(Paragraph())
```


"""

struct Paragraph <: Block end

FastAI.checkblock(::Paragraph, ::String) = true
function FastAI.mockblock(::Paragraph)
    randstring(" ABCEEFGHIJKLMNOPQESRUVWXYZ 1234567890 abcdefghijklmnopqrstynwxyz\n\t.,",
               rand(10:40))
end
