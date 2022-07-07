# Tribuo Frequently Asked Questions

## General

### Why is it called Tribuo?

"Tribuo" comes from the Latin for "to assign" or "apportion", which makes sense
since Tribuo is a prediction system for assigning outputs to examples.  Plus we
know a Latin teacher whom we'd like to keep employed.

### When did the project start?

The initial version of Tribuo was written in 2016, and the internal v1.0 was
released in the fall of 2016. The first open source release was v4.0, released
in August 2020.  Tribuo was designed after the [Oracle
Labs](https://labs.oracle.com) Machine Learning Research Group had written
several text classification projects in Java and realised the need for a good
single node ML framework on the JVM.

### What's it being used for?

Several internal groups at Oracle are using Tribuo to build ML features, many
focused around its text classification and sequence prediction features.
We're releasing it to the wider Java community to help build the ML ecosystem
on the Java platform.

### What's the license?

Tribuo is released under the Apache 2.0 license.

### How do I contribute?

We welcome bug reports, bug fixes, features, documentation improvements and
other feedback on our GitHub repo. The Tribuo mailing list is
[tribuo-devel@oss.oracle.com](mailto:tribuo-devel@oss.oracle.com), archived
[here](https://oss.oracle.com/pipermail/tribuo-devel/). We're looking at
different options for real time chat. Code contributions are accepted under the
terms of the [Oracle Contributor
Agreement](https://oca.opensource.oracle.com/).
Contributors must have sign the agreement before their PRs can be reviewed or
merged.

We're interested in how the community is using Tribuo. Our users' feedback and
feature requests have always driven Tribuo's development internally at Oracle,
and we want to continue that tradition with the open source project.

### What's the versioning strategy & compatibility plan?

Tribuo approximates semantic versioning. Major version bumps can break the
backwards compatibility of both the code and serialized models (though we hope
to fix the latter by moving to a new serialization architecture). Provided that
it's an upwards compatible change, minor version bumps can add new features,
improve performance (both statistically and in terms of runtime/memory usage),
and add new functionality to existing algorithms. Patch releases fix bugs in
existing versions and resolve security issues when they are discovered. Patch
releases may also add small methods or classes if they are required to fix
bugs.

Tribuo's dependencies may change in each type of release, but patch releases
can only bump the versions of existing dependencies (to newer patch releases of
those dependencies), and minor releases can only add new dependencies and bump
the versions of existing ones.

Anything considered part of the internal API (e.g., the innards of the tree
builders and the classes in `impl` packages outside of Core) can change in any
version, but these are usually marked in the javadoc as internal classes and
will be closed off in the module system when we adopt it.

## Project Overview 

### Why is the code broken out by prediction task (e.g., Classification, Regression, etc.)?

We designed Tribuo to be as modular as possible. Users are able to depend
exclusively on the pieces they need without additional unnecessary components
or third party dependencies. If you want to deploy a Tribuo
`RandomForestClassifier`, you only need the tribuo-classification-decision-tree
jar and its dependencies; it doesn't pull in TensorFlow, or anything else.
This makes it simpler to package up a production deployment; there is a smaller
code surface to test, fewer jars, and less space used up with unnecessary
things.

This early design choice has lead to some additional complexity in the
development of the core Tribuo library, and we're interested to see if the
community finds this structure useful.

### Where's my fit/predict interface?

Scikit-learn has popularised the fit/predict style in Python Machine Learning
libraries, and given Python's lax approach to typing, those methods are only
part of the API by convention rather that being enforced by the type system. In
Tribuo, we've separated training from prediction. Tribuo's fit method is called
"train" and lives on the `Trainer` interface, whereas Tribuo's "predict" method
lives on the `Model` class. Tribuo uses the same predict call to produce both the
outputs and the scores for those outputs. Its predict method is the equivalent
of both "predict" and "predict\_proba" in scikit-learn. We made this separation
between training and prediction so as to enable the type system to act as a
gate-keeper on prediction; predictions cannot be made using untrained models
when it's impossible to have an untrained model with a predict method. This
separation means that integrating new libraries is more complex with Tribuo
than with scikit-learn, since to conform to the scikit-learn API it is possible 
to simply export a small number of methods with specific names. With Tribuo on 
the other hand, the library needs to depend on Tribuo; however, implementing 
Tribuo's interfaces comes with other benefits.

### Why are there feature objects? Why not just use arrays?

Primitive arrays in Java are fast, but they imply a dense feature space. One of
Tribuo's design goals is strong support for NLP tasks, which typically have
high-dimensional, sparse feature spaces.  As a result, *every* feature space in
Tribuo is implicitly sparse, unlike the implicit assumption of density made by
most ML libraries. Another consequence of supporting NLP tasks is that Tribuo's
features are *named*. Each Feature is a tuple of a String and a value. This
makes it easy to understand if there is a feature space mismatch (as commonly
occurs in NLP when there are out-of-vocabulary terms). Since a Tribuo model
knows the names of all the features, it can tell when it encounters an
unexpected feature name. This prevents the possibility of loading a model and
applying it to data from a completely different domain (i.e., applying an MNIST
model to text classification) as the feature spaces will be misaligned: not
only will there be a different number of features, but they'll have different
names, too.  Since this situation results in an absence of valid features in
the supplied Example, Tribuo's predict methods will throw a RuntimeException.

### Why are the model outputs "Predictions"? Why not use arrays?

The model's `Prediction` contains a set of *named* outputs. These names make it
easy to understand which score goes with which output. Returning an array means
the user has to manually maintain the mapping between the array index and the
name of the output (e.g., "hire" = 0, "re-interview" = 1, "reject" = 2) in a
separate location from the model file itself. This leads to bugs and mismatches
when the user loads the wrong model or uses the wrong mapping. With Tribuo's
approach *this can never happen*; the model knows what its output domain is,
and can describe it to the user in the form the user expects (i.e., Strings).

### Why don't features or outputs have id numbers?

In truth, they do, but feature ids and output ids are managed by Tribuo, and
they should never need to be seen by a user of Tribuo. These ids are
automatically generated, and should only be necessary for debugging new model
implementations or interfaces. Having the ids managed by the library ensures
that they can't be confused when chaining models together, loading data, or
featurising inputs.

### What's this about provenance?

Provenance of `Model`s, `Dataset`s and `Evaluation`s is one of the core
benefits of Tribuo.  It means each model, dataset and evaluation knows exactly
how it was created, and moreover, it can generate a configuration file that can
reconstruct the object in question from scratch (assuming you still have access
to the original training and testing data). The provenance and configuration
systems come from [OLCUT](https://github.com/oracle/olcut) (Oracle Labs
Configuration and Utility Toolkit), a long lived internal library from Oracle
Labs which has roots in the configuration system used in Sphinx4. OLCUT
provides configuration files in multiple formats and includes ways to operate
on provenance in XML, JSON and protobuf format (other provenance file formats
may be added in the future).

### What's the difference between configuration and provenance?

In short, the configuration sets the parameters for an object (e.g.,
hyperparameters, data scaling, and random seed). The provenance is the
configuration plus the information gathered from the specific run that created
the model/dataset/evaluation (e.g., the number of features, the number of
samples, the timestamp of the data file, and the number of times that the
Trainer's RNG has been used).

The provenance is a superset of the configuration. You can convert a provenance
object into a set of configurations, one for each of its constituent parts. In
contrast, the configuration cannot be converted into a provenance without
executing the code (e.g., loading the dataset or training the model) as
otherwise, it won't know the run-specific information.

### How do I know what I can configure in a class?

Tribuo's configurable classes all implement 
`com.oracle.labs.mlrg.olcut.config.Configurable` and classes which implement
this can have (possibly private) fields annotated with 
`@com.oracle.labs.mlrg.olcut.config.Config` denoting they can be set by
the configuration system. Fields in superclasses can also be set by the 
configuration provided the superclass also implements `Configurable`.
You can find out what fields are configurable either by inspecting the source
on [GitHub](https://github.com/oracle/tribuo) or by running the 
`com.oracle.labs.mlrg.olcut.config.DescribeConfigurable` utility and handing it
a configurable class name. This utility can produce an example configuration 
snippet in any of OLCUT's supported configuration languages. For more
details see Tribuo's configuration tutorial.

### What's the difference between a DataSource and a Dataset?

A `DataSource` performs the inbound ETL step from the source data on disk or
from a database.  It's responsible for featurising the data (e.g., converting
text into bigram counts), reading the ground truth outputs, and creating the
`Example`s to contain the features and outputs. A `DataSource` can be lazy; it
doesn't require that all examples be in memory at once (although in practice
many of the implementations do load all of the examples). A `Dataset`, on the
other hand, is something suitable for training a model. It has the full feature
domain and the full output domain. It keeps every training example in memory
and can be split into training and testing chunks in a repeatable way.
`Dataset`s can also be transformed using the statistics of the data.  For
example, `Dataset`s can be rescaled so that the features are constrained
between zero and one. These transformations are recorded in the `Dataset` so
that they can be recovered via provenance or incorporated into a
`TransformedModel` that applies the transformations to each input before
prediction.

### What's the purpose of `Output.fullEquals`?

The `Output.equals` and `Output.hashcode` methods are constrained to only look
at the dimension labels. This means that two `Label`s can be compared for
equality even if they have different confidence scores (as ground truth labels
usually have an undefined confidence score, and predicted ones have a defined
one). To compare the values including any confidence score the
`Output.fullEquals` method should be used.  Note, this implementation of equals
and hashcode causes any two `Regressor`s that share the same dimension names to
be equal, which is unfortunate. When comparing `Regressor`, always use
`Regressor.fullEquals` to include both the regressed value, and the variance.
As `Regressor` uses `Double.NaN` as the sentinel value to indicate that no
variance was calculated, NaN variances are considered equal to each other.
