<p align="center"><img width="50%" alt="Tribuo Logo" src="docs/img/Tribuo_Logo_Colour.png" /></p>

# Tribuo - A Java prediction library (v4.0)

[Tribuo](https://tribuo.org) is a machine learning library in Java, providing
multi-class classification, regression, clustering, anomaly detection and
multi-label classification. It provides implementations of popular ML
algorithms and also wraps other libraries to provide a unified interface.
Tribuo contains all the code necessary to load, featurise and transform data. 
Additionally, it includes the evaluation classes for all supported prediction 
types. Development is led by [Oracle Labs'](https://labs.oracle.com) Machine 
Learning Research Group and we welcome community contributions.

All the trainers are configurable using the
[OLCUT](https://github.com/oracle/olcut) configuration system. This allows a
user to define a trainer in an xml file, and repeatably build models. There are
example configurations for each of the supplied Trainers in the config folder
of each package. These configuration files can also be written in json or edn
by using the appropriate OLCUT configuration dependency. Models are
serializable using Java serialization, as are the datasets themselves. 

All models and evaluations include a serializable provenance object which
records when the model or evaluation was created, what data was used, any
transformations applied to the data, the hyperparameters of the trainer, and
for evaluations, what model was used. This information can be extracted out
into JSON, or can be serialised directly using Java serialisation. For
production deployments this provenance information can be redacted and replaced
with a hash to provide model tracking through an external system.

Tribuo runs on Java 8+, and we test on LTS versions of Java, along with the
latest release.  Tribuo itself is a pure Java library and supported on all Java
platforms, however some of our interfaces require native code, and those are
supported only where the native library is. We test on x86\_64 architectures on
Windows 10, macOS and Linux (RHEL/OL/CentOS 7+), as these are supported
platforms for the native libraries we interface with. If you're interested in
another platform and wish to use one of the native library interfaces (ONNX
Runtime, TensorFlow, and XGBoost) then we recommend reaching out to the
developers of those libraries.

## Documentation

* [Library Architecture](docs/Architecture.md)
* [Package Overview](docs/PackageOverview.md)
* [Javadoc](https://tribuo.org/javadoc/4.0.0/index.html)
* [Developer Documentation](docs/Internals.md)
* [Roadmap](docs/Roadmap.md)
* [Frequently Asked Questions](docs/FAQs.md)

## Tutorials

We have tutorial notebooks for Classification, Clustering, Regression, Anomaly
Detection and the configuration system in [tutorials](tutorials). These use the
[IJava](https://github.com/SpencerPark/IJava) Jupyter notebook kernel, and work
with Java 10+. The code in the tutorials should be straightforwardly
convertible back to Java 8 code by replacing the `var` keyword with the
appropriate types.

## Algorithms

### General predictors

Tribuo has several implementations which can be used for multiple prediction tasks:

|Algorithm|Implementation|Notes|
|---|---|---|
|Bagging|Tribuo|Can use any Tribuo trainer as the base learner|
|Random Forest|Tribuo|Can use any Tribuo tree trainer as the base learner|
|K-NN|Tribuo|Has several parallel backends, as well as a single threaded backend|
|Neural Networks|TensorFlow|Via the TensorFlow interface. Models can be deployed using the ONNX interface or the TF interface|

The ensembles and K-NN use a combination function to produce the output,
those combiners are prediction task specific but the ensemble & K-NN implementations
are task agnostic. We provide voting and averaging combiners for classification and regression tasks.

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

Tribuo also has a linear chain CRF for sequence classification tasks. This is also
trained via SGD using any of Tribuo's gradient optimizers.

### Regression

Tribuo's regression algorithms are multidimensional by default, any single dimensional implementations are wrapped
so they can produce a multidimensional output.

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

Tribuo has infrastructure for clustering and a single algorithm. We expect to add new implementations over time.

|Algorithm|Implementation|Notes|
|---|---|---|
|K-Means|Tribuo|Has both sequential and parallel backends|

### Anomaly Detection

Tribuo has infrastructure for anomaly detection tasks and a single backend implementation using LibSVM.
We expect to add new implementations over time.

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

Binaries are on Maven Central, using groupId `org.tribuo`. To pull in all of
Tribuo, including the bindings for TensorFlow, ONNX Runtime and XGBoost (which
are native libraries), with Maven use:
```xml
<dependency>
    <groupId>org.tribuo</groupId>
    <artifactId>tribuo-all</artifactId>
    <version>4.0.0</version>
</dependency>
```
or from Gradle:
```groovy
api group: 'org.tribuo', name: 'tribuo-all', version: '4.0.0'
```

Most of Tribuo is pure Java and thus cross-platform, however some of the
interfaces link to libraries which use native code. Those interfaces
(TensorFlow, ONNX Runtime and XGBoost) only run on supported platforms for the
respective published binaries, and Tribuo has no control over which binaries
are supplied. If you need support for a specific platform, reach out to the
maintainers of those projects.

Individual jars are published for each Tribuo module, and it's preferred to
depend only on the modules necessary for the specific project to prevent
pulling in large dependencies like TensorFlow.

## Compiling from source

Tribuo uses [Apache Maven](https://maven.apache.org/) v3.5 or higher to build.
It's compatible with Java 8+, and we test on LTS versions of Java, along with
the latest release. To build simply run `mvn clean package`. All Tribuo's
dependencies should be on Maven Central, please file an issue for build related
issues if you're having trouble (though do check if you're missing proxy
settings for Maven first, as that's a common cause of build failures, and out
of our control).

## Contributing

We welcome contributions! See our [contribution guidelines](./CONTRIBUTING.md).

We have a discussion mailing list
[tribuo-devel@oss.oracle.com](mailto:tribuo-devel@oss.oracle.com), archived
[here](https://oss.oracle.com/pipermail/tribuo-devel/), and a [Slack
community](). For bug reports, feature requests or other issues, please file a
[Github Issue](https://github.com/oracle/tribuo/issues).

Security issues should follow our [reporting guidelines](./SECURITY.md).

## License

Tribuo is licensed under the [Apache 2.0 License](./LICENSE.txt).

## Release Notes:

- v4.0.0 - Initial public release.
- v3 - Added provenance system, the external model support and onnx integrations.
- v2 - Expanded beyond a classification system, to support regression, clustering and multi-label classification.
- v1 - Initial internal release. This release only supported multi-class classification.
