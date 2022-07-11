
const CONTEXTS = (Training(), Validation(), Inference())

"""
    checktask_core(task, sample, model; device = identity)
    checktask_core(task; device = identity)

Check if `task` conforms to the [core interface](docs/interfaces/core.md).
`sample` and `model` are used for testing. If you have implemented the testing
interface and don't supply these as arguments, `mocksample(task)` and
`mockmodel(task)` will be used.
"""
function checktask_core(task;
                        model = mockmodel(task),
                        sample = mocksample(task),
                        devicefn = identity)
    Test.@testset "Core interface" begin
        Test.@testset "`encode`" begin for context in CONTEXTS
            Test.@test_nowarn encodesample(task, context, sample)
        end end
        Test.@testset "Model compatibility" begin for context in CONTEXTS
            x, _ = encodesample(task, context, sample)
            Test.@test_nowarn ŷ = _predictx(task, model, x, devicefn)
        end end
        Test.@testset "`decodeypred" begin for context in CONTEXTS
            x, _ = encodesample(task, context, sample)
            ŷ = _predictx(task, model, x, devicefn)
            Test.@test_nowarn decodeypred(task, context, ŷ)
        end end
    end
end

function _predictx(method, model, x, device = identity)
    if shouldbatch(method)
        x = MLUtils.batch([x])
    end
    ŷs = device(model)(device(x))
    if shouldbatch(method)
        ŷ = ŷs[((:) for _ in 1:(ndims(ŷs) - 1))..., 1]
    end
    return ŷ
end
