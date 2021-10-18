# Defines detection for default `ShowBackend` and convenience methods
# that use that backend.


"""
    default_showbackend()

Return the default [`ShowBackend`](#) to use. If a Makie.jl backend
is loaded (i.e. `Makie.current_backend[] !== missing`), return [`ShowMakie`](#).
Else, return [`ShowText`](#).
"""
default_showbackend() = ShowText()


showblock(block, data) = showblock(default_showbackend(), block, data)
showblocks(block, datas) = showblocks(default_showbackend(), block, datas)
