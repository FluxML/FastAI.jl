# Defines detection for default `ShowBackend` and convenience methods
# that use that backend.


"""
    default_showbackend()

Return the default [`ShowBackend`](#) to use. If a Makie.jl backend
is loaded (i.e. `Makie.current_backend[] !== missing`), return [`ShowMakie`](#).
Else, return [`ShowText`](#).
"""
default_showbackend() = ShowText()


showblock(block, obs) = showblock(default_showbackend(), block, obs)
showblocks(block, obss) = showblocks(default_showbackend(), block, obss)
showblockinterpretable(encodings, block, obs) =
    showblockinterpretable(default_showbackend(), encodings, block, obs)
showblocksinterpretable(encodings, block, obs) =
    showblocksinterpretable(default_showbackend(), encodings, block, obs)
