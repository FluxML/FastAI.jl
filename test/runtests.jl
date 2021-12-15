




##
using ReTest
FastAI.runtests([ReTest.fail, ReTest.not(ReTest.pass)])

module FastAITests

using InlineTest

include("imports.jl")


include("encodingapi.jl")
include("fasterai.jl")
include("learner.jl")
include("training/paramgroups.jl")
include("training/discriminativelrs.jl")
include("training/fitonecycle.jl")
include("training/finetune.jl")
include("training/lrfind.jl")

include("makie.jl")

end

FastAITests.runtests([ReTest.fail, ReTest.not(ReTest.pass)])
