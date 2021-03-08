
_parenttfm(t::Transform) = t
_parenttfm(t::BufferedThreadsafe) = _parenttfm(t.buffereds[1])
_parenttfm(t::Buffered) = _parenttfm(t.tfm)


_copyrec(t::Tuple) = Tuple(_copyrec(x) for x in t)
_copyrec(x) = copy(x)

_copyrec!(dstt::Tuple, srct::Tuple) = Tuple(_copyrec!(dst, src) for (dst, src) in zip(dstt, srct))
_copyrec!(dst, src) = copy!(dst, src)
