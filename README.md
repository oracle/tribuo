<p align="center"><img width="50%" alt="Tribuo Logo" src="docs/img/Tribuo_Logo_Colour.png" /></p>

# Tribuo - A Java prediction library (v4.0)

[Tribuo](https://tribuo.org) is a machine learning library in Java that
provides multi-class classification, regression, clustering, anomaly detection
and multi-label classification. Tribuo provides implementations of popular ML
algorithms and also wraps other libraries to provide a unified interface.
Tribuo contains all the code necessary to load, featurise and transform data.
Additionally, it includes the evaluation classes for all supported prediction
types. Development is led by [Oracle Labs](https://labs.oracle.com)' Machine
Learning Research Group;  we welcome community contributions.

All trainers are configurable using the
[OLCUT](https://github.com/oracle/olcut) configuration system. This allows a
user to define a trainer in an xml file and repeatably build models.  Example
configurations for each of the supplied Trainers can be found in the config
folder of each package. These configuration files can also be written in json
or edn by using the appropriate OLCUT configuration dependency. Models and
datasets are serializable using Java serialization. 

All models and evaluations include a serializable provenance object which
records the creation time of the model or evaluation, the identity of the data
and any transformations applied to it, as well as the hyperparameters of the
trainer. In the case of evaluations, this provenance information also includes
the specific model used. Provenance information can be extracted as JSON, or
serialised directly using Java serialisation. For production deployments,
provenance information can be redacted and replaced with a hash to provide
model tracking through an external system.

Tribuo runs on Java 8+, and we test on LTS versions of Java along with the
latest release.  Tribuo itself is a pure Java library and is supported on all
Java platforms;  however, some of our interfaces require native code and are
thus supported only where there is native library support. We test on x86\_64
architectures on Windows 10, macOS and Linux (RHEL/OL/CentOS 7+), as these are
supported platforms for the native libraries with which we interface. If you're
interested in another platform and wish to use one of the native library
interfaces (ONNX Runtime, TensorFlow, and XGBoost), we recommend reaching out
to the developers of those libraries.

## Documentation

* [Library Architecture](docs/Architecture.md)
* [Package Overview](docs/PackageOverview.md)
* [Javadoc](https://tribuo.org/javadoc/4.0.0/index.html)
* [Helper Programs](docs/HelperPrograms.md)
* [Developer Documentation](docs/Internals.md)
* [Roadmap](docs/Roadmap.md)
* [Frequently Asked Questions](docs/FAQs.md)

## Tutorials

Tutorial notebooks, including examples of Classification, Clustering,
Regression, Anomaly Detection and the configuration system, can be found in the
[tutorials](tutorials). These use the [IJava](https://github.com/SpencerPark/IJava) 
Jupyter notebook kernel, and work with Java 10+. To convert
the tutorials' code back to Java 8, simply replace the `var` keyword with the
appropriate types.

## Algorithms

### General predictors

Tribuo includes implementations of several algorithms suitable for a wide range 
of prediction tasks:

|Algorithm|Implementation|Notes|
|---|---|---|
|Bagging|Tribuo|Can use any Tribuo trainer as the base learner|
|Random Forest|Tribuo|Can use any Tribuo tree trainer as the base learner|
|K-NN|Tribuo|Includes options for several parallel backends, as well as a single threaded backend|
|Neural Networks|TensorFlow|Pass a TensorFlow Neural Net to a Tribuo wrapper. Models can be deployed using the ONNX interface or the TF interface|

The ensembles and K-NN use a combination function to produce their output.
These combiners are prediction task specific, but the ensemble & K-NN 
implementations are task agnostic. We provide voting and averaging combiners
for classification and regression tasks.

### Classification

Tribuo has implementations or interfaces for:

|Algorithm|Implementation|Notes|
|---|---|---|
|Linear models|Tribuo|Uses SGD and allows any gradient optimizer|
|CART|Tribuo||
|SVM-SGD|Tribuo|An implementation of the Pegasos algorithm|
|Adaboost.SAMME|Tribuo|Can use any Tribuo classification trainer as the base learner|
|Multinomial Naive Bayes|Tribuo|
|LIME|Tribuo|Our LIME implementation allows mixing of text and tabular data, but does not support images||
|Regularised Linear Models|LibLinear||
|SVM|LibSVM or LibLinear|LibLinear only supports linear SVMs|
|Gradient Boosted Decision Trees|XGBoost||

Tribuo also supplies a linear chain CRF for sequence classification tasks. This
CRF is trained via SGD using any of Tribuo's gradient optimizers.

### Regression

Tribuo's regression algorithms are multidimensional by default. Single 
dimensional implementations are wrapped in order to produce multidimensional
output.

|Algorithm|Implementation|Notes|
|---|---|---|
|Linear models|Tribuo|Uses SGD and allows any gradient optimizer|
|CART|Tribuo||
|Lasso|Tribuo|Using the LARS algorithm|
|Elastic Net|Tribuo|Using the co-ordinate descent algorithm|
|Regularised Linear Models|LibLinear||
|SVM|LibSVM or LibLinear|LibLinear only supports linear SVMs|
|Gradient Boosted Decision Trees|XGBoost||

### Clustering

Tribuo includes infrastructure for clustering and also supplies a single
clustering algorithm implementation. We expect to implement additional
algorithms over time.

|Algorithm|Implementation|Notes|
|---|---|---|
|K-Means|Tribuo|Includes both sequential and parallel backends|

### Anomaly Detection

Tribuo offers infrastructure for anomaly detection tasks and a single backend 
implementation using LibSVM. We expect to add new implementations over time.

|Algorithm|Implementation|Notes|
|---|---|---|
|One-class SVM|LibSVM||

### Interfaces

In addition to our own implementations of Machine Learning algorithms, Tribuo
also provides a common interface to popular ML tools on the JVM. If you're
interested in contributing a new interface, open a GitHub Issue, and we can
discuss how it would fit into Tribuo.

Currently we have interfaces to:

* [LibLinear](https://github.com/bwaldvogel/liblinear-java) - via the LibLinear-java port of the original [LibLinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/).
* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - using the pure Java transformed version of the C++ implementation.
* [ONNX Runtime](https://onnxruntime.ai) - via the Java API contributed by our group.
* [TensorFlow](https://tensorflow.org) - Using the 1.14 Java API. We're participating in the Tensorflow JVM SIG, 
and the upcoming TensorFlow 2 Java API will support training models without Python, which we'll incorporate into Tribuo 
when it's released.
* [XGBoost](https://xgboost.ai)

## Binaries

Binaries are available on Maven Central, using groupId `org.tribuo`. To pull
all of Tribuo, including the bindings for TensorFlow, ONNX Runtime and XGBoost
(which are native libraries), use:

Maven:
```xml
<dependency>
    <groupId>org.tribuo</groupId>
    <artifactId>tribuo-all</artifactId>
    <version>4.0.0</version>
    <type>pom</type>
</dependency>
```
or from Gradle:
```groovy
api 'org.tribuo:tribuo-all:4.0.0@pom'
```

The `tribuo-all` dependency is a pom which depends on all the Tribuo
subprojects.

Most of Tribuo is pure Java and thus cross-platform, however some of the
interfaces link to libraries which use native code. Those interfaces
(TensorFlow, ONNX Runtime and XGBoost) only run on supported platforms for the
respective published binaries, and Tribuo has no control over which binaries
are supplied. If you need support for a specific platform, reach out to the
maintainers of those projects.

Individual jars are published for each Tribuo module. It is preferable to
depend only on the modules necessary for the specific project. This prevents
your code from unnecessarily pulling in large dependencies like TensorFlow

## Compiling from source

Tribuo uses [Apache Maven](https://maven.apache.org/) v3.5 or higher to build.
Tribuo is compatible with Java 8+, and we test on LTS versions of Java along
with the latest release. To build, simply run `mvn clean package`. All Tribuo's
dependencies should be available on Maven Central. Please file an issue for
build-related issues if you're having trouble (though do check if you're
missing proxy settings for Maven first, as that's a common cause of build
failures, and out of our control).

## Contributing

We welcome contributions! See our [contribution guidelines](./CONTRIBUTING.md).

We have a discussion mailing list
[tribuo-devel@oss.oracle.com](mailto:tribuo-devel@oss.oracle.com), archived
[here](https://oss.oracle.com/pipermail/tribuo-devel/). We're investigating
different options for real time chat, check back in the near future. For bug
reports, feature requests or other issues, please file a [Github
Issue](https://github.com/oracle/tribuo/issues).

Security issues should follow our [reporting guidelines](./SECURITY.md).

## License

Tribuo is licensed under the [Apache 2.0 License](./LICENSE.txt).

## Release Notes:

- v4.0.0 - Initial public release.
- v3 - Added provenance system, the external model support and onnx integrations.
- v2 - Expanded beyond a classification system, to support regression, clustering and multi-label classification.
- v1 - Initial internal release. This release only supported multi-class classification.
