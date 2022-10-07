# Tribuo v4.3 Release Notes

Tribuo v4.3 adds feature selection for classification problems, support for
guided generation of model cards, and protobuf serialization for all
serializable classes.  In addition there is a new interface for distance based
computations which can now use a kd-tree or brute force comparisons, the sparse
linear model package has been rewritten to use Tribuo's linear algebra system
improving the speed and reducing memory consumption, and we've added some more
tutorials.

Note this is likely the last feature release of Tribuo to support Java 8. The
next major version of Tribuo will require Java 17. In addition, support for
using `java.io.Serializable` for serialization will be removed in the next
major release, and Tribuo will exclusively use protobuf based serialization.

## Feature Selection

In this release we've added support for feature selection algorithms to the
dataset and provenance systems, along with implementations of 4 information
theoretic feature selection algorithms for use in classification problems.  The
algorithms (MIM, CMIM, mRMR and JMI) are described in this [review
paper](https://jmlr.org/papers/v13/brown12a.html). Continuous inputs are
discretised into a fixed number of equal width bins before the mutual
information is computed. These algorithms are a useful feature selection
baseline, and we welcome contributions to extend the set of supported
algorithms.

- Feature selection algorithms [#254](https://github.com/oracle/tribuo/pull/254).

## Model Card Support

[Model Cards](https://dl.acm.org/doi/10.1145/3287560.3287596) are a popular way
of describing a model, its training data, expected applications and any use
cases that should be avoided. In this release we've added guided generation of
model cards, where many fields are automatically generated from the provenance
information inside each Tribuo model. Fields which require user input (such as
the expected use cases for a model, or its license) can be added via a CLI
program, and the resulting model card can be saved in json format.

At the moment, the automatic data extraction fails on some kinds of nested
ensemble models which are generated without using a Tribuo `Trainer` class,
in the future we'll look at improving the data extraction for this case.

- Model card infrastructure ([#243](https://github.com/oracle/tribuo/pull/243), [#250](https://github.com/oracle/tribuo/pull/250), [#253](https://github.com/oracle/tribuo/pull/253)).

## Protobuf Serialization

In this release we've added [protocol
buffer](https://developers.google.com/protocol-buffers) definitions for
serializing all of Tribuo's serializable types, along with the necessary code
to interact with those definitions. This effort has improved the validation of
serialized data, and will allow Tribuo models to be upwards compatible across
major versions of Tribuo. Any serialized model or dataset from Tribuo v4.2 or
earlier can be loaded in and saved out into the new format which will ensure
compatibility with the next major version of Tribuo.

- Protobuf support for core types ([#226](https://github.com/oracle/tribuo/pull/226), [#255](https://github.com/oracle/tribuo/pull/255), [#262](https://github.com/oracle/tribuo/pull/262), [#264](https://github.com/oracle/tribuo/pull/264)).
- Protobuf support for models (Multinomial Naive Bayes [#267](https://github.com/oracle/tribuo/pull/267), Sparse linear models [#269](https://github.com/oracle/tribuo/pull/269), XGBoost [#270](https://github.com/oracle/tribuo/pull/270), OCI, ONNX and TF [#271](https://github.com/oracle/tribuo/pull/271), LibSVM [#272](https://github.com/oracle/tribuo/pull/272), LibLinear [#273](https://github.com/oracle/tribuo/pull/273), SGD [#275](https://github.com/oracle/tribuo/pull/275), Clustering models [#276](https://github.com/oracle/tribuo/pull/276), Baseline models and ensembles [#277](https://github.com/oracle/tribuo/pull/277), Trees [#278](https://github.com/oracle/tribuo/pull/278)).
- Docs and supporting programs ([#279](https://github.com/oracle/tribuo/pull/279)).

## Smaller improvements

We added an interface for querying the nearest neighbours of a vector, and
updated HDBSCAN, K-Means and K-NN to use the new interface. The old
implementation has been renamed the "brute force" search operator, and a new
implementation which uses a kd-tree has been added.

- Distance refactor ([#213](https://github.com/oracle/tribuo/pull/213), [#216](https://github.com/oracle/tribuo/pull/216), [#221](https://github.com/oracle/tribuo/pull/221), [#231](https://github.com/oracle/tribuo/pull/231), [#285](https://github.com/oracle/tribuo/pull/285)).

We migrated off Apache Commons Math, which necessitated adding several methods
to Tribuo's math library. In the process we refactored the sparse linear model
code, removing redundant matrix operations and greatly improving the speed of
LASSO.

- Refactor sparse linear models and remove Apache Commons Math ([#241](https://github.com/oracle/tribuo/pull/241)).

The ONNX export support has been refactored to allow the use of different ONNX
opsets, and custom ONNX operations. This allows users of Tribuo's ONNX export
support to supply their own operations, and increases the flexibility of the
ONNX support on the JVM.

- ONNX operator refactor ([#245](https://github.com/oracle/tribuo/pull/245)).

ONNX Runtime has been upgraded to v1.12.1, which includes Linux ARM64 and macOS
ARM64 binaries. As a result we've removed the ONNX tests from the arm Maven
profile, and so those tests will execute on Linux & macOS ARM64 platforms.

- ONNX Runtime upgrade ([#256](https://github.com/oracle/tribuo/pull/256)).

## Small improvements

- Improved the assignment to the noise cluster in HDBSCAN ([#222](https://github.com/oracle/tribuo/pull/222)).
- Upgrade liblinear-java to v2.44 ([#228](https://github.com/oracle/tribuo/pull/228)).
- Added accessors for the HDBSCAN cluster exemplars ([#229](https://github.com/oracle/tribuo/pull/229)).
- Improve validation of salts when hashing feature names ([#237](https://github.com/oracle/tribuo/pull/237)).
- Added accessors to TransformedModel for the wrapped model ([#244](https://github.com/oracle/tribuo/pull/244)).
- Added a regex text preprocessor ([#247](https://github.com/oracle/tribuo/pull/247)).
- Upgrade OpenCSV to v5.6 ([#259](https://github.com/oracle/tribuo/pull/259)).
- Added a builder to RowProcessor to make it less confusing ([#263](https://github.com/oracle/tribuo/pull/263)).
- Upgrade TF-Java to v0.4.2 ([#281](https://github.com/oracle/tribuo/pull/281)).
- Upgrade OCI Java SDK to v2.46.0, protobuf-java to 3.19.6, XGBoost to 1.6.2, jackson to 2.14.0-rc1 ([#288](https://github.com/oracle/tribuo/pull/288)).

## Bug Fixes

- Fix for HDBSCAN small cluster generation ([#236](https://github.com/oracle/tribuo/pull/236)).
- XGBoost provenance capture ([#239](https://github.com/oracle/tribuo/pull/239).

## Contributors

- Adam Pocock ([@Craigacp](https://github.com/Craigacp))
- Jack Sullivan ([@JackSullivan](https://github.com/JackSullivan))
- Romina Mahinpei ([@rmahinpei](https://github.com/rmahinpei))
- Philip Ogren ([@pogren](https://github.com/pogren))
- Katie Younglove ([@katieyounglove](https://github.com/katieyounglove))
- Jeffrey Alexander ([@jhalexand](https://github.com/jhalexand))
- Geoff Stewart ([@geoffreydstewart](https://github.com/geoffreydstewart))

