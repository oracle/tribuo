# Helper Programs

Tribuo contains a number of small programs which come in three forms:
exploratory command shells for examining model or dataset metadata, small
utilities for processing models and datasets, and example programs showing
train/test loops with configuration.

Most of these programs interact with Java serialization in some way, and so
should be secured using the appropriate JEP 290 allowlist. For more information
on Tribuo's use of Java serialization see the [security docs](./Security.md). 
All programs have been updated to read the new protobuf based Tribuo model format.

All of these programs are built using OLCUT's command line arguments
processing, and so print out the available arguments when ran with `--usage` or
`--help`. See the specific help text for each program to discover its
arguments.

## Shells

The shell applications use an [OLCUT](https://github.com/oracle/olcut) command
shell to provide an interactive CLI for exploring a serialised object.

### ModelExplorer

Found in tribuo-core, `org.tribuo.ModelExplorer` provides a shell for
inspecting trained Tribuo Models. It allows you to inspect the feature and
output domains of the loaded model, display its provenance, to view what the
model considers to be its top features, and to calculate simple statistics
like the number of features which occurred more than a specified number of
times in the training dataset.

### SequenceModelExplorer

Found in tribuo-core, `org.tribuo.sequence.SequenceModelExplorer` performs all
the same operations as [ModelExplorer](#ModelExplorer), but on SequenceModel
rather than Model.

### DatasetExplorer

Found in tribuo-data, `org.tribuo.data.DatasetExplorer` provides a shell for
inspecting serialized Tribuo Datasets. It allows you to query the feature and
output domains, view the number of features, outputs and examples, view the
Dataset's provenance, and to save out the Dataset as a csv file.

### LIMETextCLI

Found in tribuo-classification-explanations,
`org.tribuo.classification.explanations.lime.LIMETextCLI` provides a shell for
generating LIME explanations from text problems, using the default text
processing pipeline. In most cases this pipeline will not be the one used to
build a model, and so this program will require modification to be suitable for
any specific model. However it shows the overall flow of the LIME explainer,
and provides a reasonable structure for building a CLI tool with the
appropriate text processing built in.

## Utilities

These utility programs exist mainly for convenience and perform simple
operations best done in user code if they are required, however we consider
`StripProvenance` part of the supported API as it performs a complex function
and is best expressed as a standalone program.

### OCIModelCLI

Found in tribuo-oci, `org.tribuo.oci.OCIModelCLI` can deploy a Tribuo 
multi-class classification model in OCI Data Science using the model deployment
API. It provides a CLI wrapper around the functions in `org.tribuo.oci.OCIUtil`
which can deploy classification, regression and multi-label classification 
models to OCI.

### PreprocessAndSerialize

Found in tribuo-data, `org.tribuo.data.PreprocessAndSerialize` loads in a
ConfigurableDataSource from the supplied config file and writes out the
resulting Dataset as a serialized object.

### SQLToCSV

Found in tribuo-data, `org.tribuo.data.sql.SQLToCSV` executes the supplied SQL
query and writes out the result set to the specified CSV file. It does not
validate the SQL query in any way, so use it carefully.

### SplitTextData

Found in tribuo-data, `org.tribuo.data.text.SplitTextData` splits a text file
in Tribuo's default text input format (i.e., each line is of the form 
`<output> ## <input-text>`) into two files, one for training and one for 
testing. It validates each line before splitting, and logs any odd lines.

### StripProvenance

Found in tribuo-json, `org.tribuo.json.StripProvenance` can read in a
serialized model file, and write out a version without some or all of the
provenance information. It can remove the dataset, trainer or instance
provenances, or any combination of the three. Optionally it can write out the
current provenance as a json file, and it can insert a hash of the old
provenance into the new model. This hash can be used to track the new model,
and it's expected that the hash is used as a key into some storage mechanism
where the original provenance json is stored.

We consider `StripProvenance` to be an essential part of the provenance system
in Tribuo, as deploying models which are visible to third parties may
necessitate that the provenance is removed, so we need to provide a mechanism
to do so. However if you wish to write your own mechanism to remove the
provenance (perhaps to store some different hash or key, or to only remove
paths from a model) then this class should form a useful skeleton for building
such functionality.

### ModelCardCLI

Found in tribuo-modelcard, `org.tribuo.interop.modelcard.ModelCardCLI` provides
a shell for completing model cards by supplying a set of plain text fields. This
is an interactive way of building up model cards, though it can be done directly
by building the `UsageDetails` object itself along with a `ModelCard`.

### DescribeConfigurable

Found in the olcut-core artifact, 
`com.oracle.labs.mlrg.olcut.config.DescribeConfigurable` prints a description 
of a configurable class. It shows the default values for each field, whether 
each field is mandatory, whether each field should be redacted from stored 
configuration or provenance, and a short string describing each field. While 
this is part of OLCUT rather than Tribuo we mention it here as it's useful 
when working with the configuration system. In addition to describing a 
configurable class, it can also provide a configuration snippet for that class 
in any supported OLCUT config file format. Note these snippets are not 
recursive, they don't include configurations for all the fields of the class if
 those fields are subclasses of `Configurable` as it's not possible to know 
 what an appropriate instance is just from the configuration.

## Example train/test programs

Each Tribuo backend for a given prediction type includes a program called
`TrainTest`. This provides a simple way to train and test a model on a dataset
supplied on the command line. They show how to use the particular backend, what
configuration options it has, and occasionally exposes any extra information
produced by that specific model implementation (e.g., the TrainTest programs
based on LibSVM can print out the number of support vectors used). Each of
these programs can load from a specific subset of Tribuo's supported input
types, and allows moderate configuration through the command line arguments.
These programs should be considered a starting point for working with a
specific prediction backend, we do not recommend anyone use them for more than
demos.

We also provide a number of more complex train/test harnesses which we've
detailed below. Again these programs are examples of how to use Tribuo, and
suitable for simple experiments while exploring the library or developing
functionality, we do not recommend you use them as production endpoints. They
don't have the kind of input validation that is necessary for such a task.

### ConfigurableTrainTest

Found in tribuo-data, `org.tribuo.data.ConfigurableTrainTest` loads the
specified trainer from the supplied configuration file, then trains a predictor
on the supplied training data, and evaluates it on the supplied test set. The
datasets are loaded using the `org.tribuo.data.DataOptions` class, which
supports csv, libsvm, text, and serialized formats. Optionally it can perform
cross-validation, producing the mean and standard deviation of the appropriate
evaluation metrics.

### CompletelyConfigurableTrainTest

Found in tribuo-data, `org.tribuo.data.CompletelyConfigurableTrainTest` is like
[ConfigurableTrainTest](#ConfigurableTrainTest), but expects the training and
testing datasources to be configurable, and loaded from the config file. This
program can be used to run a complete experiment only using the configuration
system and a few flags.

### tribuo-classification-experiments' ConfigurableTrainTest

Found in tribuo-classification-experiments,
`org.tribuo.classification.experiments.ConfigurableTrainTest` is similar to the
one from [data](#ConfigurableTrainTest) but is specialised to classification
problems, and so requires fewer arguments and less configuration. It also 
exposes classification specific options such as per label weights for training, 
and also the ability to write out the predictions to a text file for downstream
processing.

### RunAll

Found in tribuo-classification-experiments,
`org.tribuo.classification.experiments.RunAll` performs the same train/test
experiment on every Trainer in the supplied configuration file. This can be
used to quickly compare several trainers, either instances of the same trainer
with different hyperparameters, or different trainers. It writes out the model
and the performance evaluations to files in the specified directory.

### Test

Found in tribuo-classification-experiments,
`org.tribuo.classification.experiments.Test` loads in a serialised model file,
and evaluates it on the specified testing dataset, optionally writing out the
predictions to a text file.

### tribuo-classification-experiments' TrainTest

Found in tribuo-classification-experiments,
`org.tribuo.classification.experiments.TrainTest` aggregates most of the
classification algorithms available in Tribuo into a single program. A single
program with a huge number of command line arguments. This lets you easily run
a variety of different algorithms from the same program to see their
performance.

### SeqTest

Found in tribuo-classification-sgd, `org.tribuo.classification.sgd.crf.SeqTest`
lets you train a CRF model using the specified gradient descent algorithm on
either a toy sequence dataset, or a serialised `SequenceDataset` specified as
an argument. This serves as an example of how to train and test a sequence
model, though in general sequences are evaluated using a more complex
methodology than the one used in `LabelSequenceEvaluator`.
