# Architecture

Tribuo is a library for making Machine Learning models, and then using those
models to make predictions on previously unseen data. 

A ML model is the result of applying some training algorithm to a dataset, 
producing what's usually a large number of floating point values, but could be 
a tree structured if/else statement among many other things. In Tribuo a model
is that, along with the necessary feature and output statistics to map from
the named feature space into Tribuo's ids, and to map Tribuo's output ids into
the named output space.

## Data flow overview
<p align="center"><img width="75%" alt="Tribuo architecture diagram" src="img/tribuo-data-flow.png" /></p>

Tribuo loads in data from disk or a DB using a `DataSource` implementation.
This `DataSource` processes the input data, converting it into Tribuo's storage
format, an `Example`. An `Example` is a tuple of an `Output` (i.e. what you
want to predict) and a list of `Feature`s, where each `Feature` is a tuple of a
`String` feature name and a `double` feature value.  The `DataSource` is then
read into a `Dataset`, which accumulates statistics about the data for future
use in model construction. `Dataset`s can be split into chunks to separate out
training and testing data, or to filter out examples according to some
criterion. Once the `Dataset` has been processed, it's passed to a `Trainer`,
which contains the training algorithm along with any necessary parameter values
(in ML these are called hyperparameters to differentiate them from Model
parameters which are learned), and the `Trainer` performs some iterations of
the training algorithm before producing the `Model`. `Model`s contain the
necessary parameters to make predictions along with a `Provenance` object which
records how the `Model` was constructed (e.g. data file name, data hash,
trainer hyperparameters, time stamp, etc).  Both `Model`s and `Dataset`s can be
serialized out to disk using Java Serialization. Once a model has been trained
it can be fed previously unseen `Example`s and produces `Prediction`s for the
`Output`s of those examples. If the new `Example`s have known `Output`s they
can be passed to an `Evaluator` which calculates statistics like the accuracy
(i.e. the number of times the predicted output was the same as the provided
output).
