# Tribuo Frequently Asked Questions

## General

### Why is it called Tribuo?

It's from the latin to assign or apportion, and it's a prediction system for
assigning outputs to examples. Plus we know a latin teacher, and we'd like to
keep them employed.

### When did the project start?

The initial version of Tribuo was written in 2016 with the internal v1.0
release in the fall of 2016. The first open source release was v4.0, released
in July 2020.  Tribuo was designed after the Machine Learning Research Group in
[Oracle Labs](https://labs.oracle.com) had written several text classification
projects in Java, and realised the need for a good single node ML framework on
the JVM.

### What's it being used for?

We have several internal groups at Oracle building ML features using Tribuo,
many focused around it's text classification and sequence prediction features.
We're releasing it to the wider Java community to help build the ML ecosystem on
the Java platform.

### What's the license?

Tribuo is released under the Apache 2.0 license.

### How do I contribute?

We welcome bug reports, bug fixes, features, documentation improvements and
other feedback on our GitHub repo. The Tribuo mailing list is
[tribuo-devel@oss.oracle.com](mailto:tribuo-devel@oss.oracle.com), archived 
[here](https://oss.oracle.com/pipermail/tribuo-devel/),and we have a 
[Slack community](). Code contributions are accepted under the terms of the [Oracle
Contributor Agreement](https://www.oracle.com/technetwork/community/oca-486395.html), 
and contributors must have signed it before their PRs can be reviewed or merged.

We're also interested in how people are using Tribuo, our users' feedback and
feature requests has always driven it's development internally at Oracle, and
we want to continue that with the open source project.

## Project Overview 

### Why is the code broken out by prediction task (e.g. Classification, Regression etc)?

We designed Tribuo to be as modular as possible, with users able to depend on
only the pieces they need without additional unnecessary components or third
party dependencies. If you want to deploy a Tribuo Random Forest, you only need
the tribuo-classification-decision-tree jar and it's dependencies, it doesn't
pull in TensorFlow, or anything else. This makes it simpler to package up a 
production deployment, as there is a smaller code surface to test, fewer jars
and less space used up with unnecessary things.

This early design choice has lead to some additional complexity in the
development of the core Tribuo library, and we're interested to see if the
community finds this structure useful.

### Where's my fit/predict interface?

Scikit-learn has popularised the fit/predict style in Python Machine Learning
libraries, and given Python's lax approach to typing, those methods are only
part of the API by convention rather that being enforced by the type system. In
Tribuo we've split out training from predction, so Tribuo's fit method is
called "train" and lives on the Trainer interface, and Tribuo's predict lives
on the Model class. Tribuo uses the same predict call to produce both the
outputs and the scores for those outputs, it's predict method is the equivalent
of both "predict" and "predict\_proba" in scikit-learn. We made this separation
to use the type system to enforce that predictions are only made with trained
models, as it's not possible to have an untrained model with a predict method.
This does mean that integrating new libraries is more complex than
scikit-learn, as the interface needs to depend on Tribuo, rather than export a
couple of methods with specific names, but implementing Tribuo's interfaces
comes with other benefits.

### Why are there feature objects? Why not just use arrays?

Primitive arrays in Java are definitely faster, but they imply a dense feature
space. One of Tribuo's design goals was strong support for NLP tasks, which
typically have high-dimensional, sparse feature spaces.  As a result, *every*
feature space in Tribuo is implicitly sparse (unlike the implict dense
assumption in most ML libraries). Another consequence is that Tribuo's features
are *named*, each Feature is a tuple of a String and a value. This makes it
easy to understand if there is a feature space mismatch (as commonly occurs in
NLP when there are out of vocabulary terms), as the Tribuo model knows the
names of all the features, and can tell when there is a new name it's not seen
before. This means that it's not possible to load a model in and apply it to
data from a completely different domain (i.e. applying an MNIST model to text
classification) as the feature spaces won't line up, not only will there be a
different number of features, but they'll all have different names too. In such
a situation, Tribuo's predict methods will throw a RuntimeException as there
are no valid features in the supplied Example.

### Why are the model outputs "Predictions"? Why not use arrays?

The model's prediction contains a set of outputs that had positive scores, and
each output is *named*, making it easy to understand which score goes with
which output. Returning an array means the user has to maintain the mapping
between the array index and the name of the output (e.g. "hire" = 0,
"re-interview" = 1, "reject" = 2) manually, separate from the model file
itself. This leads to bugs and mismatches when the user loads the wrong model
or uses the wrong mapping. With Tribuo's approach *this can never happen*, the
model knows what it's output domain is, and can describe it to the user in the
form the user expects (i.e. Strings).

### Why don't features or outputs have id numbers?

In truth, they do, but feature ids and output ids are managed by Tribuo, and
should never need to be seen by a user of Tribuo. Those ids are automatically
generated, and should only be necessary for debugging new model implementations
or interfaces. Having the ids managed by the library ensures that they can't be
confused when chaining models together, or during data loading or
featurisation.

### What's this about provenance?

Provenance (of Models, Datasets and Evaluations) is one of the core benefits of
Tribuo.  It means each model, dataset and evaluation knows exactly how it was
created, and moreover it can generate a configuration file which can
reconstruct the object in question from scratch (assuming you still have access
to the original training and testing data). The provenance and configuration
systems come from [OLCUT](https://github.com/oracle/olcut) (Oracle Labs
Configuration and Utility Toolkit), a long lived internal library from Oracle
Labs which has roots in the configuration system used in Sphinx4. OLCUT provides
configuration files in multiple formats, and ways to operate on provenance in
JSON format (other provenance file formats will be added in the future).

### What's the difference between configuration and provenance?

In short, configuration is the parameter settings for an object (e.g.
hyperparameters, data scaling, random seed), and provenance is the
configuration plus information gathered from the specific run that created the
model/dataset/evaluation (e.g. the number of features, the number of samples,
the timestamp of the data, the number of times that Trainer's RNG had been
used).

Provenance is a superset of configuration, and you can convert a provenance
object into configurations for all it's constituent parts, but you can't
convert a configuration into a provenance without executing the code (e.g.
loading the dataset, training the model) as otherwise it won't know the
run-specific information.

### What's the difference between a DataSource and a Dataset?

A DataSource does the inbound ETL step from the source data on disk or in a
database.  It's responsible for featurising the data (e.g. converting text into
bigram counts), reading the ground truth outputs, and creating the Examples to
contain the features and outputs. A DataSource can be lazy, it doesn't require
that all the examples are in memory at once (though in practice many of the
implementations do load in everything). A Dataset is something suitable for
training a model, it has the full feature domain, the full output domain, keeps
every training example in memory, and can be split into training and testing
chunks in a repeatable way. Datasets can also be transformed, e.g. rescaling
the features to be between zero and one, and other operations which require
using statistics of all the data. These transformations are recorded in the
Dataset so they can be recovered via provenance or incorporated into a 
TransformedModel that applies the transformations to each input before
prediction.
