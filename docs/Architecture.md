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
criterion. As `Example`s are fed into a `Dataset` the `Feature`s are observed
and have their statistics recorded in a `FeatureMap`. Similarly the `Output`s 
are recorded in the appropriate `OutputInfo` subclass for the specified `Output`
subclass. Once the `Dataset` has been processed, it's passed to a `Trainer`,
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

## Structure

Tribuo has several top level modules:
- Core provides Tribuo's core classes and interfaces.
- Data provides loaders for text, sql and csv data, along with the columnar package which provides infrastructure for
working with columnar data.
- Math provides Tribuo's linear algebra library, along with kernels and gradient optimizers.
- Json provides a json data loader, and a tool to strip provenance from trained models.

Tribuo has separate modules for each prediction task:
- Classification contains an `Output` implementation called `Label` which represents a multi-class classification. 
Each `Label` is a tuple of a String name, and a double precision score value. The Classification package has companion
implementations of `OutputFactory`, `OutputInfo`, `Evaluator` and `Evaluation` called `LabelFactory`, `LabelInfo`,
`LabelEvaluator` and `LabelEvaluation` respectively.
- Regression contains an `Output` implementation called `Regressor` which represents multidimensional regression.
Each `Regressor` is a tuple of dimension names, double precision dimension values, and double precision dimension 
variances. It has companion implementations of `OutputFactory`, `OutputInfo`, `Evaluator` and `Evaluation` called
`RegressionFactory`, `RegressionInfo`, `RegressionEvaluator` and `RegressionEvaluation` respectively. By default
the dimensions are named "DIM-x" where x is a non-negative integer.
- AnomalyDetection contains an `Output` implementation called `Event` which represents the detection of an 
anomalous or expected event (represented by the `EventType` enum containing `ANOMALY` and `EXPECTED`). Each `Event` is
a tuple of an `EventType` instance and a double precision score value, representing the score of the event type. The
AnomalyDetection package has companion implementations of `OutputFactory`, `OutputInfo`, `Evaluator` and `Evaluation`
called `AnomalyFactory`, `AnomalyInfo`, `AnomalyEvaluator` and `AnomalyEvaluation` respectively. 
- Clustering contains an `Output` implementation called `ClusterID` which represents the cluster id number assigned.
Each `ClusterID` is a non-negative integer id number, and a double precision score representing the strength of association.
The Clustering package has companion implementations of `OutputFactory`, `OutputInfo`, `Evaluator` and 
`Evaluation` called `ClusteringFactory`, `ClusteringInfo`, `ClusteringEvaluator` and `ClusteringEvaluation` respectively.
- MultiLabel contains an `Output` implementation called `MultiLabel` which represents a multi-label classification.
Each `MultiLabel` is a possibly empty set of `Label` instances with their associated scores.
The MultiLabel package has companion implementations of `OutputFactory`, `OutputInfo`, `Evaluator` and `Evaluation` 
called `MultiLabelFactory`, `MultiLabelInfo`, `MultiLabelEvaluator` and `MultiLabelEvaluation` respectively. It 
also has a `Trainer` which accepts a `Trainer<Label>` and generates a `Model<Label>` by using the inner trainer to
make independent predictions for each `Label`. This is a reasonable baseline strategy to use for multi-label problems.

Finally there are cross cutting module collections:
- Common provides shared infrastructure for the prediction tasks.
- Interop provides infrastructure for working with large external libraries like TensorFlow and ONNX Runtime.
- Util provides independent libraries that Tribuo uses for certain tasks: InformationTheory is a library
of information theoretic functions, and Tokens provides the interface Tribuo uses for tokenization along with 
implementations of several tokenizers.