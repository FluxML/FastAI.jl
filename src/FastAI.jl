module FastAI

using Base: NamedTuple
using Reexport
@reexport using FluxTraining
import MLUtils
using MLUtils: getobs, numobs, splitobs, eachobs, DataLoader
using Flux

using FilePathsBase
using Flux
using Flux.Optimise
import Flux.Optimise: apply!, Optimiser, WeightDecay
using FluxTraining: Learner, handle
using FluxTraining.Events
using JLD2: jldsave, jldopen
using Markdown
using PrettyTables
using Requires
using Setfield
import Test
import UnicodePlots
using Statistics
using InlineTest


# ## Learning task API (previously DLPipelines.jl)
include("tasks/task.jl")
include("tasks/taskdata.jl")
include("tasks/predict.jl")
include("tasks/check.jl")

# ## Data block API
include("datablock/block.jl")
include("datablock/encoding.jl")
include("datablock/task.jl")
include("datablock/describe.jl")
include("datablock/wrappers.jl")

# ## Blocks
# ### Wrapper blocks
include("blocks/many.jl")

# ### Other
include("blocks/continuous.jl")
include("blocks/label.jl")

# ## Encodings
# ### Wrapper encodings
include("encodings/only.jl")

# ### Other
include("encodings/onehot.jl")

# Training interface
include("datablock/models.jl")
include("datablock/loss.jl")

# Interpretation
include("interpretation/backend.jl")
include("interpretation/text.jl")
include("interpretation/makie.jl")
include("interpretation/task.jl")
include("interpretation/showinterpretable.jl")
include("interpretation/learner.jl")
include("interpretation/detect.jl")

# Training
include("learner.jl")
include("training/paramgroups.jl")
include("training/discriminativelrs.jl")
include("training/utils.jl")
include("training/onecycle.jl")
include("training/finetune.jl")
include("training/lrfind.jl")
include("training/metrics.jl")

include("serialization.jl")

# submodules
include("datasets/Datasets.jl")
@reexport using .Datasets

include("Registries/Registries.jl")
@reexport using .Registries

export
# submodules
      Datasets,
      Models,
      getobs,
      numobs,
      mapobs, eachobs, groupobs, shuffleobs, splitobs, ObsView,

# task API
      taskmodel,
      taskdataset,
      taskdataloaders,
      tasklossfn,
      encodesample,
      predict,
      predictbatch,
      Training,
      Validation,
      Inference,
      Context,

# blocks
      Label,
      LabelMulti,
      Many,
      TableRow,
      Continuous,
      Paragraph,

# encodings
      encode,
      decode,
      setup,
      OneHot,
      Only,
      Named,
      SupervisedTask,
      BlockTask,
      describetask,
      checkblock,
      makebatch,
      getbatch,

# interpretation
      ShowText,
      ShowMakie,
      showblock,
      showblocks,
      showsample,
      showsamples,
      showoutput,
      showoutputs,
      showoutputbatch,
      showencodedsample,
      showencodedsamples,
      showbatch,
      showprediction,
      showpredictions,

# training
      tasklearner,
      Learner,
      fit!,
      fitonecycle!,
      finetune!,
      lrfind,
      savetaskmodel,
      loadtaskmodel,
      accuracy_thresh, gpu,
      plot

end  # module
