# Architecture

Tribuo is a library for making Machine Learning models, and then using those
models to make predictions on previously unseen data. 

A ML model is the result of applying some training algorithm to a dataset, 
producing what's usually a large number of floating point values, but could be 
a tree structured if/else statement among many other things. In Tribuo a model
is that, along with the necessary feature and output statistics to map from
the named feature space into Tribuo's ids, and to map Tribuo's output ids into
the named output space.

Another way to think about a Tribuo `Model` is a learned mapping from a 
*sparse* feature space of doubles into a *dense* output space (e.g. of class 
label probabilities, or regressed outputs etc). Every dimension, both input and
output, is named so these names can be used to check that the input and the 
model agree on the feature space they are using.

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

## Configuration, Options and Provenance

Many of Tribuo's trainers, datasources and other classes implement the `Configurable` interface. This is 
provided by [OLCUT](https://github.com/oracle/olcut), and allows for runtime configuration of classes based
on configuration files written in a variety of formats (the default format is xml, and json & edn are available).

The configuration system is integrated into the command line arguments `Options` system build into OLCUT's 
`ConfigurationManager`. Values in configuration files can be overridden on the command line by supplying
`--@<object-name>.<field-name> <value>` in the arguments. The configuration system provides the basis of Tribuo's 
model tracking `Provenance` system, which records all hyperparameters, dataset parameters (e.g. file location, 
train/test split etc), and any user supplied instance information, along with run specific information such as the
file hash, number of training examples etc. A model provenance can be converted into a list of configurations 
for each `Configurable` object involved in the model training. Similarly an evaluation provenance can be converted
into the configurations for the model, and the configurations for the test dataset. These configurations can be
loaded into a fresh `ConfigurationManager`, optionally saved to disk, and the evaluation or model training can be 
repeated (or rerun with tweaks like new data or a hyperparameter change).

Configurable classes have `@Config` annotations on their fields, and such fields have the value from the 
configuration file inserted into them upon construction in the configuration system. A snippet from the classification
SGD trainer is given below to illustrate this:

```java
public class LinearSGDTrainer implements Trainer<Label>, WeightedExamples {
    @Config(description="The classification objective function to use.")
    private LabelObjective objective = new LogMulticlass();

    @Config(description="The gradient optimiser to use.")
    private StochasticGradientOptimiser optimiser = new AdaGrad(1.0,0.1);

    @Config(description="The number of gradient descent epochs.")
    private int epochs = 5;

    @Config(description="Log values after this many updates.")
    private int loggingInterval = -1;

    @Config(description="Minibatch size in SGD.")
    private int minibatchSize = 1;

    @Config(description="Seed for the RNG used to shuffle elements.")
    private long seed = Trainer.DEFAULT_SEED;

    @Config(description="Shuffle the data before each epoch. Only turn off for debugging.")
    private boolean shuffle = true;

    private SplittableRandom rng;

    private int trainInvocationCounter;
}
```

Only fields which are configured need to be annotated `@Config`, other fields can be set in the appropriate constructor.
OLCUT requires that all classes which implement `Configurable` have a no-args constructor, and the interface allows for
a `postConfig` method, which is called after the object has been constructed and had the appropriate field values
inserted, but before it is published or returned from the `ConfigurationManager`. This `postConfig` method is used
for the same validation that you would perform in a constructor, and can be called from the regular constructors.
Default values for the configurable parameters can be specified in the way that default fields are usually specified,
and the `@Config` annotation has optional parameters for the description, if the field is mandatory, and if the
field value should be redacted from any configuration or provenance based on this object. More details about OLCUT
can be found in it's [documentation](https://github.com/oracle/OLCUT).

The `LinearSGDTrainer` class above is configured by the xml snippet below:

```xml
<config> 
   <component name="logistic" type="org.tribuo.classification.sgd.linear.LinearSGDTrainer">
        <property name="objective" value="log"/>
        <property name="optimiser" value="adam"/>
        <property name="epochs" value="10"/>
        <property name="loggingInterval" value="100"/>
        <property name="minibatchSize" value="1"/>
        <property name="seed" value="1"/>
    </component>
    <component name="log" type="org.tribuo.classification.sgd.objectives.LogMulticlass"/>
    <component name="adam" type="org.tribuo.math.optimisers.Adam">
        <property name="initialLearningRate" value="3e-4"/>
        <property name="betaOne" value="0.95"/>
    </component>
</config>
```

which instantiates `LinearSGDTrainer` using a logistic regression objective, with the `Adam` gradient optimiser 
using [Andrei Karpathy's preferred learning rate](https://twitter.com/karpathy/status/801621764144971776) and an 
adjusted beta one parameter (note these parameters are just demonstration values, we're not recommending these 
specific values).

## Data Loading

### Built-in formats

Tribuo supports several common input formats for loading in data:
- libsvm/svmlight - a sparse numerical format for classification and regression tasks.
- CSV - a plain text delimited format (using an RFC4180 compliant parser).
- JSON - JavaScript Object Notation, Tribuo natively reads JSON objects which are a map from String to primitive value,
and the whole file is an array of such objects.
- SQL - Tribuo has a JDBC loader which can query a database and convert the result set into Tribuo `Example`s.
- text - a one document per line format, with the response variable before the text delimited by ` ## `.

There are two CSV loaders, a simple one for reading a CSV file (with or without a header) where all the columns are
either features or responses, and a complex loader based on Tribuo's `RowProcessor`. The `RowProcessor` also underlies
the SQL and JSON loaders, and is extremely configurable. For more details see the [Columnar Inputs](#columnar-inputs)
section below. If there are other common formats that are of interest, let us know by filing an issue.

Tribuo's interfaces are extensible, and implementing another format simply requires implementing the `DataSource` 
interface. We recommend looking at `LibSVMDataSource` or `TextDataSource` to see how to implement one for a flat
file format. For columnar data, Tribuo has specialised processing infrastructure. This is used for the CSV, JSON and SQL
loaders, and provides a large amount of flexibility.

### Columnar Inputs

Columnar data sources require a configurable extraction step to map the columns into Tribuo `Example` and `Feature` 
objects. This is because a single column may emit many features, or none at all, some columns may be unnecessary, 
some may form `Example` level metadata, and the output variable needs to be specified. To support this 
usecase Tribuo provides the `RowProcessor` a configurable mechanism for converting a `ColumnarIterator.Row`, which
is a tuple of a `Map<String,String>` and an row number into an `Example`. The `RowProcessor` uses 4 interfaces to 
process the input map:
- `FieldExtractor` - processes the whole row at once, extracting metadata fields which are written into the `Example`. 
Metadata fields are things like the `Example`'s id number. The `Example`'s weight is handled as a special case of the
metadata processing as described in the javadoc.
- `FieldProcessor` - processes a single field, producing a (possibly empty) list of `Feature`s.
- `FeatureProcessor` - processes all the features after they have been generated by a `FieldProcessor`. This allows the
generation of features which depend upon multiple other features (such as conjunctions), and also to filter out
irrelevant or unnecessary features.
- `ResponseProcessor` - processes the designated response fields, using the supplied `OutputFactory` to convert 
the field text into an `Output` instance.

The different interfaces are supplied to the `RowProcessor` on construction (or configuration). By default 
`FieldProcessor`s are bound to a single column, but there is an optional system which generates new `FieldProcessor`s
based on supplied regexes. This can be used if the data is drawn from a schema-less format, where the user doesn't
know what fields will be present in each document, or if the set of fields is large and the number of unique
`FieldProcessor`s is small (e.g. so that the same field processor can be applied to all columns beginning with "A", 
without writing a very large configuration or code file to describe them all). These regexes are usually instantiated
once, before any rows are processed, but `RowProcessor` is intentionally subclass-able so developers can trigger
expansion whenever necessary. In the current implementation there is at most one `FieldProcessor` per field, we'll
reconsider this restriction if there is sufficient interest.

Internally the `RowProcessor` operates on `ColumnarFeature` which is a feature subclass that tracks both the feature
name and the column name. It's used to allow additional flexibility in the `FeatureProcessor`s when generating 
conjunction or other cross-cutting features. The `Example` contract does not guarantee that the feature
objects are preserved after being stored in an `Example` so don't depend on `ColumnarFeature` outside of the columnar
processing infrastructure.

If your columnar data is not in a format currently supported by Tribuo, you can subclass `ColumnarDataSource`, provide
an implementation of `ColumnarIterator` which converts from your input format into a `ColumnarIterator.Row` which is a
`Map<String,String>` and an index representing the row number, and then configure the `RowProcessor` to extract
`Example`s from your data.

### Splitting up Datasets

`DataSource`s are not designed for splitting data into chunks, but Tribuo provides several mechanisms to split up
datasets to provide training and test splits, subsample data based on it's properties, or to create cross validation
folds. The train/test and cross-validation splits are self explanatory, though it's worth nothing that the 
cross-validation splits use the feature domain of the underlying dataset. The `DatasetView` underlies the 
cross-validation splits and can also be constructed using a predicate function (or a list of indices). The predicate 
function accepts an `Example` and so can depend on the features, outputs or metadata encoded in an `Example`.

## Transforming datasets

Tribuo supports independent feature based transformations, i.e. operations like rescaling or binning features.
This uses the `org.tribuo.transform` package, which provides the mechanisms for fitting and applying transformations.
Transformations can be chained and are applied in the supplied sequence to the specified feature. After the local
transformations, a transformation chain can be applied to each feature in turn (called the global transformation).
Similar to the `RowProcessor` described above the transformations can also be applied to a regex, and every feature 
name which matches the regex has a copy of the transformation pipeline instantiated and applied to that feature. There
is validation to ensure that a regex transformation cannot apply to a feature that already has a specific local 
transformation chain, if this occurs it'll throw an exception. These transformations are also applied to the 
feature domain to ensure it maintains the proper statistics.

We plan to introduce global feature transformations in some future release, to allow operations like PCA or other
feature extraction steps.

## Obfuscation

One of Tribuo's benefits is it's extensive tracking of model metadata and provenance, however we realise this metadata
isn't necessarily something that should live in deployed models that third parties have access to. As a result Tribuo
provides a few transformation mechanisms to remove metadata from a trained model.

### Provenance

Provenance can be removed from the `Model` objects using the `StripProvenance` program in the JSON module. It's
possible to remove the three kinds of stored provenance separately: trainer provenance, data provenance, instance 
provenance. Also the SHA-256 hash of the full provenance object can be inserted into the object as a tracking mechanism,
we intend that the user stores the hash as a key for the original provenance JSON in an external storage mechanism.
Alternatively `@Config` fields can be marked `redact=True` which will prevent those values from being stored in
the provenance or any configuration.

### Feature Hashing

In addition to its use as a dimensionality reduction technique, feature hashing also obfuscates the original feature
names provided the system doesn't store the forward mapping. Tribuo provides an implementation of feature hashing
that lives entirely in the feature domain object to avoid storing the forward mapping from original names to
hashed names. This means that Tribuo has no knowledge of the true feature names, and the system transparently hashes 
the inputs. The feature names tend to be particularly sensitive when working with NLP problems, as for example bigrams 
would otherwise appear in the feature domains.
