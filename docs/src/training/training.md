# Training

```@contents
Pages = ["training/training.md"]
```

<!-- A copy of what's in the fastai doc. -->
<!-- The fastai library structures its training process around the Learner class, whose object binds together a PyTorch model, a dataset, an optimizer, and a loss function; the entire learner object then will allow us to launch training.

basic_train defines this Learner class, along with the wrapper around the PyTorch optimizer that the library uses. It defines the basic training loop that is used each time you call the fit method (or one of its variants) in fastai. This training loop is very bare-bones and has very few lines of codes; you can customize it by supplying an optional Callback argument to the fit method.

callback defines the Callback class and the CallbackHandler class that is responsible for the communication between the training loop and the Callback's methods. The CallbackHandler maintains a state dictionary able to provide each Callback object all the information of the training loop it belongs to, putting any imaginable tweaks of the training loop within your reach.

callbacks implements each predefined Callback class of the fastai library in a separate module. Some modules deal with scheduling the hyperparameters, like callbacks.one_cycle, callbacks.lr_finder and callback.general_sched. Others allow special kinds of training like callbacks.fp16 (mixed precision) and callbacks.rnn. The Recorder and callbacks.hooks are useful to save some internal data generated in the training loop.

train then uses these callbacks to implement useful helper functions. Lastly, metrics contains all the functions and classes you might want to use to evaluate your training results; simpler metrics are implemented as functions while more complicated ones as subclasses of Callback. For more details on implementing metrics as Callback, please refer to creating your own metrics. -->