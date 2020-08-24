#=
metric.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the FastAI Metric API to Julia

Definition of the metrics that can be used in training models

This code is inspired by FastAI, but differs from it in important ways

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/metrics.py

The documentation is copied from here

https://dev.fast.ai/metrics
=#

"""
    AbstractMetric

An abstract metric.

# Required interface
- `reset!(metric::T)`: Reset inner state to prepare for new computation
- `accumulate!(metric::T, values)`: Update the state with new results
- `value(metric::T)`: The value of the metric
- `name(metric::T)`: Name of the metric, camel-cased and with "Metric" removed
"""
abstract type AbstractMetric end

"""
    implements_metric(T::DataType)

Test if `T` implements the [`AbstractMetric`](@ref) interface.
"""
function implements_metric(T::DataType)
    return hasmethod(reset!,(T,)) &&
        hasmethod(accumulate!,(T,Any)) &&
        hasmethod(value,(T,)) &&
        hasmethod(name,(T,))
end

"""
    AverageFunctionMetric <: AbstractMetric
    AverageFunctionMetric(func)

Average the values of `func` taking into account potentially different batch sizes.
"""
mutable struct AverageFunctionMetric <: AbstractMetric
    func
    total::Float64
    count::Int
end
AverageFunctionMetric(func) = AverageFunctionMetric(func, 0.0, 0)

function reset!(metric::AverageFunctionMetric)
    metric.total = 0.0
    metric.count = 0
end

function accumulate!(metric::AverageFunctionMetric, values)
    bs = length(values)
    metric.total += metric.func(values)*bs
    metric.count += bs
end

value(metric::AverageFunctionMetric) = if metric.count > 0 metric.total/metric.count else nothing end

name(metric::AverageFunctionMetric) = "Avg$(metric.func)"

@assert implements_metric(AverageFunctionMetric)

"""
    AvgLoss <: AbstractMetric
    AvgLoss()

Average the losses taking into account potentiall different batch sizes.
"""
mutable struct AverageMetric <: AbstractMetric
    total:: Real
    count:: Int
end
AverageMetric() = AverageMetric(0.0, 0)

reset!(al::AverageMetric) = al.total, al.count = 0.0, 0

function accumulate!(al::AverageMetric, values...)
    batch_loss, batch_size = values 
    al.total += batch_loss
    al.count += batch_size
end 

value(al::AverageMetric) = if al.count>0 al.total/al.count else nothing end

name(al::AverageMetric) = "Average"

@assert implements_metric(AverageMetric)

"""
    SmoothMetric <: AbstractMetric
    SmoothMetric(alpha = 0.98)

`alpha`-exponential smoothing of values.
"""
mutable struct SmoothMetric <: AbstractMetric
    alpha::Real
    val::Real
    first::Bool
end
SmoothMetric(alpha = 0.98) = SmoothMetric(alpha, 0.0, true)

reset!(asl::SmoothMetric) = asl.first=true
    
function accumulate!(asl::SmoothMetric, value)
    if asl.first
        asl.first = false
        asl.val = value
    else
        asl.val = asl.alpha*asl.val+(1-asl.alpha)*value
    end
end 

value(asl::SmoothMetric) = asl.val

name(asl::SmoothMetric) = "SmoothMetric"

@assert implements_metric(SmoothMetric)
#=
"""
class ValueMetric[source]
ValueMetric(func, metric_name=None) :: Metric

Use to include a pre-calculated metric value (for insance calculated in a Callback) and returned by func

def metric_value_fn(): return 5e-3

vm = ValueMetric(metric_value_fn, 'custom_value_metric')
test_eq(vm.value, 5e-3)
test_eq(vm.name, 'custom_value_metric')

vm = ValueMetric(metric_value_fn)
test_eq(vm.name, 'metric_value_fn')
"""
struct ValueMetric
    metric_value_fn
    metric_name::String
end

"""
AccumMetric

Stores predictions and targets on CPU in accumulate to perform 
final calculations with func.

func is only applied to the accumulated predictions/targets when 
the value attribute is asked for (so at the end of a validation/training 
phase, in use with Learner and its Recorder).The signature of func 
should be inp,targ (where inp are the predictions of the model and 
targ the corresponding labels).

For classification problems with single label, predictions need to 
be transformed with a sofmax then an argmax before being compared to 
the targets. Since a softmax doesn't change the order of the numbers, 
we can just apply the argmax. Pass along dim_argmax to have this done 
by AccumMetric (usually -1 will work pretty well). If you need to pass 
to your metrics the probabilities and not the predictions, use softmax=True.

For classification problems with multiple labels, or if your targets 
are onehot-encoded, predictions may need to pass through a sigmoid 
(if it wasn't included in your model) then be compared to a given 
threshold (to decide between 0 and 1), this is done by AccumMetric 
if you pass sigmoid=True and/or a value for thresh.

If you want to use a metric function sklearn.metrics, you will need 
to convert predictions and labels to numpy arrays with to_np=True. 
Also, scikit-learn metrics adopt the convention y_true, y_preds 
which is the opposite from us, so you will need to pass 
invert_arg=True to make AccumMetric do the inversion for you.

    Example

class AccumMetric(Metric):
    "Stores predictions and targets on CPU in accumulate to perform final calculations with `func`."
    def __init__(self, func, dim_argmax=None, activation=ActivationType.No, thresh=None, to_np=False,
                 invert_arg=False, flatten=True, **kwargs):
        store_attr(self,'func,dim_argmax,activation,thresh,flatten')
        self.to_np,self.invert_args,self.kwargs = to_np,invert_arg,kwargs

    def reset(self): self.targs,self.preds = [],[]

    def accumulate(self, learn):
        pred = learn.pred
        if self.activation in [ActivationType.Softmax, ActivationType.BinarySoftmax]:
            pred = F.softmax(pred, dim=self.dim_argmax)
            if self.activation == ActivationType.BinarySoftmax: pred = pred[:, -1]
        elif self.activation == ActivationType.Sigmoid: pred = torch.sigmoid(pred)
        elif self.dim_argmax: pred = pred.argmax(dim=self.dim_argmax)
        if self.thresh:  pred = (pred >= self.thresh)
        targ = learn.y
        pred,targ = to_detach(pred),to_detach(targ)
        if self.flatten: pred,targ = flatten_check(pred,targ)
        self.preds.append(pred)
        self.targs.append(targ)

    @property
    def value(self):
        if len(self.preds) == 0: return
        preds,targs = torch.cat(self.preds),torch.cat(self.targs)
        if self.to_np: preds,targs = preds.numpy(),targs.numpy()
        return self.func(targs, preds, **self.kwargs) if self.invert_args else self.func(preds, targs, **self.kwargs)

    @property
    def name(self):  return self.func.func.__name__ if hasattr(self.func, 'func') else  self.func.__name__    
"""
struct AccumMetric
    func
    dim_argmax
    activation:: Symbol 
    thresh
    flatten:: Bool
end

AccumMetric(
    func, 
    dim_argmax=nothing, 
    activation= :no, 
    thresh=nothing, 
    flatten=true) = AccumMetric(func, dim_argmax, activation, thresh, flatten)

value(m::AccumMetric) = 0

Compute accuracy with targ when pred is bs * n_classes

Example
    def change_targ(targ, n, c):
        idx = torch.randperm(len(targ))[:n]
        res = targ.clone()
        for i in idx: res[i] = (res[i]+random.randint(1,c-1))%c
        return res
    x = torch.randn(4,5)
    y = x.argmax(dim=1)
    test_eq(accuracy(x,y), 1)
    y1 = change_targ(y, 2, 5)
    test_eq(accuracy(x,y1), 0.5)
    test_eq(accuracy(x.unsqueeze(1).expand(4,2,5), torch.stack([y,y1], dim=1)), 0.75)
    error_rate[source]
    error_rate(inp, targ, axis=-1)

accuracy(m::AccumMetric, inp, targ, axis=-1) = nothing

1 - accuracy

Example
    x = torch.randn(4,5)
    y = x.argmax(dim=1)
    test_eq(error_rate(x,y), 0)
    y1 = change_targ(y, 2, 5)
    test_eq(error_rate(x,y1), 0.5)
    test_eq(error_rate(x.unsqueeze(1).expand(4,2,5), torch.stack([y,y1], dim=1)), 0.25)

error_rate(m::AccumMetric, inp, targ, axis=-1) = nothing

Computes the Top-k accuracy (targ is in the top k predictions of inp)

Example
    x = torch.randn(6,5)
    y = torch.arange(0,6)
    test_eq(top_k_accuracy(x[:5],y[:5]), 1)
    test_eq(top_k_accuracy(x, y), 5/6)

top_k_accuracy(m::AccumMetric, inp, targ, k=5, axis=-1) = nothing

"""
APScoreBinary

Average Precision for single-label binary classification problems

See the scikit-learn documentation for more details.
"""
struct APScoreBinary
    axis::Int
    average::Symbol
    pos_label::Int 
    sample_weight
end

APScoreBinary(axis=-1, average=:macro, pos_label=1, sample_weight=nothing) = APScoreBinary(axis, average, pos_label, sample_weight)

"""
BalancedAccuracy

Balanced Accuracy for single-label binary classification problems

See the scikit-learn documentation for more details.
"""
struct BalancedAccuracy
    axis::Int
    sample_weight
    adjusted::Bool
end

BalancedAccuracy(axis, sample_weight, adjusted) = BalancedAccuracy(axis=-1, sample_weight=nothing, adjusted=false)

BrierScore[source]
BrierScore(axis=-1, sample_weight=None, pos_label=None)

Brier score for single-label classification problems

See the scikit-learn documentation for more details.

CohenKappa[source]
CohenKappa(axis=-1, labels=None, weights=None, sample_weight=None)

Cohen kappa for single-label classification problems

See the scikit-learn documentation for more details.

F1Score[source]
F1Score(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None)

F1 score for single-label classification problems

See the scikit-learn documentation for more details.

FBeta[source]
FBeta(beta, axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None)

FBeta score with beta for single-label classification problems

See the scikit-learn documentation for more details.

HammingLoss[source]
HammingLoss(axis=-1, sample_weight=None)

Hamming loss for single-label classification problems

See the scikit-learn documentation for more details.

Jaccard[source]
Jaccard(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None)

Jaccard score for single-label classification problems

See the scikit-learn documentation for more details.

Precision[source]
Precision(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None)

Precision for single-label classification problems

See the scikit-learn documentation for more details.

Recall[source]
Recall(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None)

Recall for single-label classification problems

See the scikit-learn documentation for more details.

RocAuc[source]
RocAuc(axis=-1, average='macro', sample_weight=None, max_fpr=None, multi_class='ovr')

Area Under the Receiver Operating Characteristic Curve for single-label multiclass classification problems

See the scikit-learn documentation for more details.

RocAucBinary[source]
RocAucBinary(axis=-1, average='macro', sample_weight=None, max_fpr=None, multi_class='raise')

Area Under the Receiver Operating Characteristic Curve for single-label binary classification problems

See the scikit-learn documentation for more details.

class Perplexity[source]
Perplexity() :: AverageMetric

Perplexity (exponential of cross-entropy loss) for Language Models

x1,x2 = torch.randn(20,5),torch.randint(0, 5, (20,))
tst = perplexity
tst.reset()
vals = [0,6,15,20]
learn = TstLearner()
for i in range(3): 
    learn.y,learn.yb = x2[vals[i]:vals[i+1]],(x2[vals[i]:vals[i+1]],)
    learn.loss = F.cross_entropy(x1[vals[i]:vals[i+1]],x2[vals[i]:vals[i+1]])
    tst.accumulate(learn)
test_close(tst.value, torch.exp(F.cross_entropy(x1,x2)))
=#
