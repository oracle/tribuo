<p align="center"><img width="50%" alt="Tribuo Logo" src="docs/img/Tribuo_Logo_Colour.png" /></p>

# Tribuo - A Java prediction library (v4.0)

[Tribuo](https://tribuo.org) is a machine learning library in Java, providing
multi-class classification, regression, clustering, anomaly detection and
multi-label classification. It provides implementations of popular ML
algorithms and also wraps other libraries to provide a unified interface.
Tribuo contains all the code necessary to load, featurise and transform data,
and also evaluation classes for all the supported prediction types. Development
is led by [Oracle Labs'](https://labs.oracle.com) Machine Learning Research
Group and we welcome community contributions.

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

### Interfaces

In addition to our own implementations of Machine Learning algorithms, Tribuo
also provides a common interface to popular ML tools on the JVM. If you're
interested in contributing a new interface, open a GitHub Issue, and we can
discuss how it would fit into Tribuo.

Currently we have interfaces to:

* [LibLinear](https://github.com/bwaldvogel/liblinear-java) - via the LibLinear-java port of the original [LibLinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/).
* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
* [ONNX Runtime](https://onnxruntime.ai) - via the Java API contributed by our group.
* [TensorFlow](https://tensorflow.org) - Using the 1.14 Java API. We're working with the Tensorflow JVM SIG, 
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
