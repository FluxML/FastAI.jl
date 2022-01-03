

"""
    invariant_datacontainer(block)

Create an `Invariants.Invariant` that checks that `data` is a data container
with observations that are valid instances of block `block`. See also
[`invariant_checkblock`](#).
"""
function invariant_datacontainer(block)
    return SequenceInvariant([
        WithContext(data -> (data,), invariant_hasmethod(nobs, (data = Any,))),
        WithContext(data -> (data, 1), invariant_hasmethod(getobs, (data = Any, idx = Int))),
        WithContext(data -> getobs(data, 1), invariant_checkblock(block)),
    ], "`data` should be a data container with block observations $(summary(block))", "")
end


function invariant_hasmethod(f, argtypes::NamedTuple)
    methodcall = "`$f("
    for (i, (name, T)) in enumerate(zip(keys(argtypes), values(argtypes)))
        if i != 1
            methodcall *= ", "
        end
        methodcall *= string(name)
        if T !== Any
            methodcall *= "::$T"
        end
    end
    methodcall *= ")`"
    return SequenceInvariant(
        [
            BooleanInvariant(
                args -> Base.hasmethod(f, typeof(args)),
                "Method $methodcall should exist",
                args -> "Expected method $methodcall to exist but it does not",
            ),
            BooleanInvariant(
                args -> _hasmethod(f, args),
                "Method $methodcall should not error",
                args -> "Expected method $methodcall not to error but it did.",
            ),
        ],
        "$methodcall should be a valid method call",
        "",
    )

end

function _hasmethod(f, args)
    try
        f(args...)
        return true
    catch e
        return false
    end
end
