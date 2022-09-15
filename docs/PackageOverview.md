# Package Structure Overview

Tribuo has a modular structure to allow minimal dependencies for any specific
deployment. We describe the overall package structure below.

## Package Description

The top level project has core modules which define the API, data interactions,
a math library, and common modules shared across prediction types.

- Core - (artifactID: `tribuo-core`, package root: `org.tribuo`) Provides the main classes and interfaces:
  - `Dataset` - A list of `Example`s plus associated feature information, such
    as the number of categories for categorical features, or the mean
    and variance in the case of real-valued features.
  - `DataSource` - A list of `Example`s processed from some other format
    and accompanied by the provenance describing the source and processing of
    these `Example`s. 
  - `Example` - An array or list of String and value pairs. The `Example` is typed
    with a subclass of Output that represents the appropriate type of response.
  - `Feature` - An immutable tuple of String and value. The String is the feature
    name, which is used as the feature's unique identifier.
  - `FeatureMap` - A map from String to `VariableInfo` objects. When immutable, it
    also contains feature id numbers, although these should be treated as an
    implementation detail and not relied upon.
  - `Model` - A class that can make predictions of a specific `Output` type.
  - `Output` - An interface denoting the type of output: regression, 
    multi-label, multi-class, clustering, or anomaly detection.
  - `OutputInfo` - An interface representing information about the output.
  - `Trainer` - A class that generates `Model`s based on a `Dataset` of a specific output type.
  - `Prediction` - A class that stores the output of a `Model` when presented
    with an `Example` for labeling. It contains scores for each of the predicted
    labels. These scores may optionally be a probability distribution.
  - `VariableInfo` - A class representing information about the feature, e.g., the 
    number of times it occurred in the dataset.
- Core contains several other packages.
  - `dataset` - `Dataset`s which provide a view on another dataset, either
    subsampling it or excluding features below a threshold.
  - `datasource` - Implementations of `DataSource` which operate on in-memory or
    simple on-disk formats.
  - `ensemble` - Base classes and interfaces for ensemble models.
  - `evaluation` - Base evaluation classes for all output types. This package
    also includes evaluation-related classes for cross-validation and train-test 
    splitting.
  - `hash` - An implementation of feature hashing which obfuscates any feature
    names that are stored in a `Model`. Hashing prevents feature names from
    leaking out of the training data.
  - `provenance` - Provenance classes for Tribuo. Provenance tracks the location
    and transformations of datasets, the parameters of trainers, and other
    useful information.
  - `sequence` - A sequence prediction API.
  - `transform` - A feature transformation package that can be applied to a full
    dataset or to individual features matched via regexes. It also contains
    wrapping trainers (trainers that wrap another trainer to provide additional
    functionality) and wrapping models to ensure that the same transformations
    are always applied at prediction time.
  - `util` - Utilities for basic operations such as for working with arrays and
    random samples.
- Data - (artifactID: `tribuo-data`, package root: `org.tribuo.data`) provides classes which deal with sampled data, columnar data, csv
  files and text inputs. The user is encouraged to provide their own text
processing infrastructure implementation, as the one here is fairly basic.
  - `columnar` - The columnar package provides many useful base classes for
    extracting features from columnar data.
  - `csv` - Builds on the columnar package and supplies infrastructure for
    working with CSV and other delimiter separated data.
  - `sql` - Builds on the columnar package and supplies infrastructure for
    working with JDBC sources.
  - `text` - Text processing infrastructure interfaces and an example
    implementation.
- Json - (artifactID: `tribuo-json`, package root: `org.tribuo.json`) provides functionality
for loading from json data sources, and for stripping provenance out of a model.
- Math - (artifactID: `tribuo-math`, package root: `org.tribuo.math`) provides a linear algebra library for working with both sparse
 and dense vectors and matrices.
  - `kernel` - a set of kernel functions for use in the SGD package (and elsewhere).
  - `la` - a linear algebra library containing functions used in the
    SGD implementation. It is not a full BLAS.
  - `optimisers` - a set of stochastic gradient descent algorithms, including `SGD`
    with Momentum, `AdaGrad`, `AdaDelta`, `RMSProp` and several others. `AdaGrad`
should be considered the default algorithm since it works best across the
widest range of linear SGD problems.
  - `util` - various util classes for working with arrays, vectors and matrices.

## Util libraries

There are 3 utility libraries which are used by Tribuo but do not depend
on other parts of it.

- InformationTheory - (artifactID: `tribuo-util-infotheory`, package root: `org.tribuo.util.infotheory`) provides discrete information theoretic functions suitable
for computing clustering metrics, feature selection and structure learning.
- ONNXExport - (artifactID: `tribuo-util-onnx`, package root: `org.tribuo.util.onnx`) provides infrastructure for building ONNX graphs from Java.
This package is suitable for use in other JVM libraries which want to write ONNX models, and provides additional type safety and usability over
directly writing the protobufs.
- Tokenization - (artifactID: `tribuo-util-tokenization`, package root: `org.tribuo.util.tokens`) provides a tokenization API suitable 
for feature extraction or information retrieval, along with several tokenizer implementations, including a wordpiece implementation
suitable for use with models like BERT.

## Multi-class Classification

Multi-class classification is the act of assigning a single label from a set of
labels to a test example.  The classification module has several submodules:

| Folder                  | ArtifactID | Package root | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-------------------------| --- | --- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Core                    | `tribuo-classification-core` | `org.tribuo.classification` | Contains an Output subclass for use with multi-class classification tasks, evaluation code for checking model performance, and an implementation of Adaboost.SAMME. It also contains simple baseline classifiers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| DecisionTree            | `tribuo-classification-tree` | `org.tribuo.classification.dtree` | An implementation of CART decision trees.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Experiments             | `tribuo-classification-experiments` | `org.tribuo.classification.experiments` | A set of main functions for training & testing models on any supported dataset. This submodule depends on all the classifiers and allows easy comparison between them. It should not be imported into other projects since it is intended purely for development and testing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Explanations            | `tribuo-classification-experiments` | `org.tribuo.classification.explanations` | An implementation of LIME for classification tasks. If you use the columnar data loader, LIME can extract more information about the feature domain and provide better explanations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| FeatureSelection        | `tribuo-classification-fs` | `org.tribuo.classification.fs` | An implementation of several information theoretic feature selection algorithms for classification problems.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| LibLinear               | `tribuo-classification-liblinear` | `org.tribuo.classification.liblinear` | A wrapper around the LibLinear-java library. This provides linear-SVMs and other l1 or l2 regularised linear classifiers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| LibSVM                  | `tribuo-classification-libsvm` | `org.tribuo.classification.libsvm` | A wrapper around the Java version of LibSVM. This provides linear & kernel SVMs with sigmoid, gaussian and polynomial kernels.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Multinomial Naive Bayes | `tribuo-classification-mnnaivebayes` | `org.tribuo.classification.mnb` | An implementation of a multinomial naive bayes classifier. Since it aims to store a compact in-memory representation of the model, it only keeps track of weights for observed feature/class pairs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| SGD                     | `tribuo-classification-sgd` | `org.tribuo.classification.sgd` | An implementation of stochastic gradient descent based classifiers. It includes a linear package for logistic regression and linear-SVM (using log and hinge losses, respectively), a kernel package for training a kernel-SVM using the Pegasos algorithm, a crf package for training a linear-chain CRF, and a fm package for training pairwise factorization machines. These implementations depend upon the stochastic gradient optimisers in the main Math package. The linear, fm, and crf packages can use any of the provided gradient optimisers, which enforce various different kinds of regularisation or convergence metrics. This is the preferred package for linear classification and for sequence classification due to the speed and scalability of the SGD approach. |
| XGBoost                 | `tribuo-classification-xgboost` | `org.tribuo.classification.xgboost` | A wrapper around the XGBoost Java API. XGBoost requires a C library accessed via JNI.  XGBoost is a scalable implementation of gradient boosted trees.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

## Multi-label Classification

Multi-label classification is the task of predicting a set of labels for a test
example rather than just a single label. 

The independent binary predictor breaks each multi-label prediction into n
binary predictions, one for each possible label.  To achieve this, the supplied
trainer takes a classification trainer and uses it to build n models, one per
label, which are then run in sequence on a test example to produce the final
multi-label output. A similar approach is used in the classifier chains to
convert a classification trainer into a multi-label trainer.

| Folder | ArtifactID | Package root | Description |
| --- | --- | --- | --- |
| Core | `tribuo-multilabel-core` | `org.tribuo.multilabel` | Contains an Output subclass for multi-label prediction, evaluation code for checking the performance of a multi-label model, and a basic implementation of independent binary predictions. It also contains implementations of Classifier Chains and Classifier Chain Ensembles, which are more powerful ensemble techniques for multi-label prediction tasks. |
| SGD | `tribuo-multilabel-sgd` | `org.tribuo.multilabel.sgd` | An implementation of stochastic gradient descent based classifiers. It includes a linear package for independent logistic regression and linear-SVM (using log and hinge losses, respectively), along with factorization machines using either loss for each output label. These implementations depend upon the stochastic gradient optimisers in the main Math package. The linear and fm packages can use any of the provided gradient optimisers, which enforce various different kinds of regularisation or convergence metrics. |

## Regression

Regression is the task of predicting real-valued outputs for a test example.
This package provides several modules:

| Folder | ArtifactID | Package root | Description |
| --- | --- | --- | --- |
| Core | `tribuo-regression-core` | `org.tribuo.regression` | contains an Output subclass for use with regression data, as well as evaluation code for checking model performance using standard regression metrics (R^2, explained variance, RMSE, and mean absolute error). The module also contains simple baseline regressions. |
| LibLinear | `tribuo-regression-liblinear` | `org.tribuo.regression.liblinear` | A wrapper around the LibLinear-java library. This provides linear-SVMs and other l1 or l2 regularised linear regressions. |
| LibSVM | `tribuo-regression-libsvm` | `org.tribuo.regression.libsvm` | A wrapper around the Java version of LibSVM. This provides linear & kernel SVRs with sigmoid, gaussian and polynomial kernels. |
| RegressionTrees | `tribuo-regression-tree` | `org.tribuo.regression.rtree` | An implementation of two types of CART regression trees. The first type builds a separate tree per output dimension, while the second type builds a single tree for all outputs. |
| SGD | `tribuo-regression-sgd` | `org.tribuo.regression.sgd` | An implementation of stochastic gradient descent for linear regression and factorization machine regression. It uses the main Math package's set of gradient optimisers, which allow for various regularisation and descent algorithms. |
| SLM | `tribuo-regression-slm` | `org.tribuo.regression.slm` | An implementation of sparse linear models. It includes a co-ordinate descent implementation of ElasticNet, a LARS implementation, a LASSO implementation using LARS, and a couple of sequential forward selection algorithms. |
| XGBoost | `tribuo-regression-xgboost` | `org.tribuo.regression.xgboost` | A wrapper around the XGBoost Java API. XGBoost requires a C library accessed via JNI. |

## Clustering

Clustering is the task of grouping input data. The clustering system
implemented is single membership -- each datapoint is assigned to one and only
one cluster. This package provides two modules:

| Folder | ArtifactID | Package root | Description |
| --- | --- | --- | --- |
| Core | `tribuo-clustering-core` | `org.tribuo.clustering` | Contains the Output subclass for use with clustering data, as well as the evaluation code for measuring clustering performance. |
| HDBSCAN | `tribuo-clustering-hdbscan` | `org.tribuo.clustering.hdbscan` | An implementation of HDBSCAN, a non-parametric density based clustering algorithm. |
| KMeans | `tribuo-clustering-kmeans` | `org.tribuo.clustering.kmeans` | An implementation of K-Means using the Java 8 Stream API for parallelisation, along with the K-Means++ initialization algorithm. |

## Anomaly Detection

Anomaly detection is the task of finding outliers or anomalies at prediction
time using a model trained on non-anomalous data.  This package provides two
modules:

| Folder | ArtifactID | Package root | Description |
| --- | --- | --- | --- |
| Core | `tribuo-anomaly-core` | `org.tribuo.anomaly` | Contains the Output subclass for use with anomaly detection data. |
| LibLinear | `tribuo-anomaly-liblinear` | `org.tribuo.anomaly.liblinear` | A wrapper around the Java version of LibLinear, which provides a one-class SVM. |
| LibSVM | `tribuo-anomaly-libsvm` | `org.tribuo.anomaly.libsvm` | A wrapper around the Java version of LibSVM, which provides a one-class SVM. |

## Common

The common module shares code across multiple prediction types. It provides
the base support for LibLinear, LibSVM, nearest neighbour, tree, and XGBoost
models. The nearest neighbour submodule is standalone, however the rest of the
submodules require the prediction specific implementation modules. The common
tree package contains the implementations of Random Forests and Extremely 
Randomized Trees (ExtraTrees).

## Third party models

Tribuo supports loading a number of third party models which were trained
outside the system (even in other programming languages) and scoring them from
Java using Tribuo's infrastructure. Currently, we support loading ONNX,
TensorFlow and XGBoost models. Additionally we support wrapping an 
[OCI Data Science](https://www.oracle.com/data-science/cloud-infrastructure-data-science.html) 
model deployment in a Tribuo model.

- OCI - Supports deploying Tribuo models to OCI Data Science, and wrapping OCI
  Data Science models in Tribuo external models to allow them to be served with 
other Tribuo models.
- ONNX - [ONNX](https://onnx.ai) (Open Neural Network eXchange) format is used
  by several deep learning systems as an export format, and there are
converters from systems like scikit-learn to the ONNX format.  Tribuo provides
a wrapper around Microsoft's [ONNX Runtime](https://onnxruntime.ai) that can
score ONNX models on both CPU and GPU platforms. ONNX support is found in the
`tribuo-onnx` artifact in the `org.tribuo.interop.onnx` package which also
provides a feature extractor that uses BERT embedding models. This package can
load Tribuo-exported ONNX models and extract the stored Tribuo provenance
objects from those models.
- TensorFlow - Tribuo supports loading [TensorFlow](https://tensorflow.org)'s
  frozen graphs and saved models and scoring them.
- XGBoost - Tribuo supports loading [XGBoost](https://xgboost.ai)
  classification and regression models.

## TensorFlow

Tribuo includes experimental support for TensorFlow-Java 0.4.0 (using
TensorFlow 2.7.0) in the `tribuo-tensorflow` artifact in the
`org.tribuo.interop.tensorflow` package. Models can be defined using
TensorFlow-Java's graph construction mechanisms, and Tribuo will manage the
gradient optimizer output function and loss function. It includes a Java
serialisation system so that all TensorFlow models can be serialised and
deserialised in the same way as other Tribuo models.  TensorFlow models run by
default on GPU if one is available and the appropriate GPU jar is on the
classpath.

This support remains experimental while the TF JVM SIG rewrites the TensorFlow
Java API.  We participate in the TensorFlow JVM SIG, and are working to improve
TensorFlow not just for Tribuo but for the Java community as a whole.

Tribuo demonstrates the TensorFlow interop by including an example config file,
several example model generation functions and protobuf for an MNIST model
graph.

## Other modules

Tribuo has a number of other modules:

|  Folder | ArtifactID | Package root | Description |
|---------| --- | --- | --- | --- |
| Json  | `tribuo-json` | `org.tribuo.json` | Contains support for reading and writing Json formatted data, along with a program for inspecting and removing provenance information from models. |
| ModelCard | `tribuo-interop-modelcard` | `org.tribuo.interop.modelcard` | Contains support for reading and writing model cards in Json format, using the provenance information in Tribuo models to guide the card construction. |
| Reproducibility | `tribuo-reproducibility` | `org.tribuo.reproducibility` | A utility for reproducing Tribuo models and datasets. |
