# Tribuo - A Java prediction library (v4.0)

[Tribuo](https://tribuo.org) is a prediction system in Java, providing multi-class classification, regression, clustering, anomaly detection and multi-label classification.
It wraps other libraries to provide a unified interface and also has several implementations of useful
algorithms. It was initially developed by [Oracle Labs'](https://labs.oracle.com)
Machine Learning Research Group.

All the trainers are configurable using the [OLCUT](https://github.com/oracle/olcut) configuration system. This allows
a user to define a trainer in an xml file, and repeatably build models. There are example configurations
for each of the supplied Trainers in the config folder of each package. Models are serializable using
Java serialization, as are the datasets themselves. These configuration files can be written in
json if the olcut-config-json dependency is added along with the Tribuo dependencies.

From v3 all models and evaulations include a serialisable provenance object which records when the model or evaluation was
created, what data was used, any transformations applied to the data, the hyperparameters of the trainer, and for evaluations, what
model was used. This information can be extracted out into JSON, or can be serialised directly using Java serialisation.

## Package Description
The top level project has core modules which define the API, data interactions, a math library, and common modules shared across prediction types.

- Core - Provides the main classes and interfaces:
  - Dataset - A list of Examples, plus associated feature information (number of categories, or mean & variance for real values).
  - DataSource - A list of Examples processed from some other format, along with provenance describing where they were loaded and how they were processed.
  - Example - An array or list of String & value pairs. This is typed with a subclass of Output representing the response for this example.
  - Feature - An immutable tuple of String and value. The String is the name of the feature, and used as the unique identifier for a given feature.
  - FeatureInfo - A class representing information about the feature, e.g. number of times it occurred in the dataset.
  - FeatureMap - A map from String to FeatureInfo objects. When immutable it also contains feature id numbers, though these should be treated as an implementation detail and not relied upon.
  - Model - A class that can make predictions of a specific Output type.
  - Output - An interface denoting the type of output, either regression, multi-label, multi-class, clustering, or anomaly detection.
  - OutputInfo - An interface representing information about the output.
  - Trainer - A class that generates Models based on a Dataset of a specific output type.
  - Prediction - A class that stores the output of a Model when presented with an Example for labelling. It contains scores for each of the predicted labels, which may optionally be a probability distribution.
- Core contains several other packages.
  - dataset - Datasets which provide a view on another dataset, either subsampling it or excluding features below a threshold.
  - datasource - Implementations of DataSource which operate on in memory or simple on disk formats.
  - ensemble - Base classes and interfaces for ensemble models.
  - evaluation - Base evaluation classes for all output types, including cross-validation and train-test splitting.
  - hash - An implementation of feature hashing which obfuscates any feature names that are stored in a Model. This prevents the name of a feature from leaking out, which excludes a type of information leakage from the training data.
  - provenance - Provenance classes for Tribuo, which track the location and transformations of datasets, the parameters of trainers and other useful information.
  - sequence - A sequence prediction API.
  - transform - A feature transformation package, which can apply to a full dataset or individual features matched via regexes. It also has wrapping trainers and models to ensure the transformations are always applied at prediction time.
  - util - Utilities for working with arrays, random samples, and other basic operations.
- Data - provides classes which deal with sampled data, columnar data, csv files and text inputs. These classes are mainly optional, and the user is encouraged to provide their own text processing infrastructure.
  - columnar - The columnar package provides many useful base classes for extracting features from columnar data.
  - csv - Builds on the columnar package and supplies infrastructure for working with CSV and other delimiter separated data.
  - sql - Builds on the columnar package and supplies infrastructure for working with JDBC sources.
  - text - An example text processsing infrastructure.
- Math - provides a linear algebra library for working with vectors and matrices both sparse and dense.
  - kernel - a set of kernel functions for use in the SGD package (and elsewhere).
  - la - a linear algebra library. It currently has functions that are used in the SGD implementation and is not a full BLAS.
  - optimisers - a set of stochastic gradient descent algorithms, including SGD + Momentum, AdaGrad, AdaDelta, RMSProp and
  several others. AdaGrad should be considered the default algorithm, it works best across the widest range of linear SGD
  problems.
  - util - various util classes for working with arrays, vectors and matrices.

## Multi-class Classification

Multi-class classification is the act of assigning a single label from a set of labels to a test example.
The classification module has several submodules:

- Core - contains an Output subclass for use with multi-class classification
  tasks, evaluation code for checking model performance, and an
  implementation of Adaboost.SAMME. Also contains simple baseline classifiers, if
  your system doesn't outperform these, it's not working.
- DecisionTree - an implementation of CART decision trees.
- Experiments - A set of main functions for training & testing models on any supported dataset. It depends on all the
  classifiers and allows easy comparison between them. This should not be imported into other projects, it's purely for
  development and testing.
- Explanations - An implementation of LIME for classification tasks. If you use the columnar data loader, LIME can
  extract more information about the feature domain and provide better explanations.
- LibLinear - a wrapper around the LibLinear-java library. This provides linear-SVMs, and other l1 or l2 regularised
  linear classifiers.
- LibSVM - a wrapper around the Java version of LibSVM. This provides linear & kernel SVMs, with sigmoid, gaussian and
  polynomial kernels.
- Multinomial Naive Bayes - an implementation of a multinomial naive bayes classifier. It aims to have a compact in
  memory representation of the model, so only has weights for observed feature/class pairs.
- SGD - an implementation of stochastic gradient descent based classifiers. It has a linear package for logistic
  regression and linear-SVM (using log and hinge losses respectively), a kernel package for training a kernel-SVM using the
  Pegasos algorithm, and a crf package for training a linear-chain CRF. These implementations depend upon the stochastic
  gradient optimisers in the main Math package. The linear and crf packages can use any of the provided gradient optimisers,
  which enforce various different kinds of regularisation or convergence metrics. This is the preferred package for linear
  classification and for sequence classification due to the speed and scalability of the SGD approach.
- XGBoost - a wrapper around the XGBoost Java API. XGBoost requires a C library accessed via JNI. 
  XGBoost is a scalable implementation of gradient boosted trees, and one of the highest accuracy classifiers
  provided in this project.

## Multi-label Classification

Multi-label classification is the task of predicting a set of labels for a test example, rather than just a single label.
Currently this package provides an Output subclass for multi-label prediction, evaluation code for checking the
performance of a multi-label model, and a basic implementation of independent binary predictions. The simplest
multi-label approach is to break each multi-label prediction into n binary predictions, one for each possible label.
The supplied trainer takes another classification trainer and uses it to build n models, one per label, which then are ran
in sequence on a test example to produce the final multi-label output.

There are many more complicated multi-label approaches in the literature that use the label structure to improve
performance, we'll consider adding these based on demand from the community.

## Regression

Regression is the task of predicting real valued outputs for a test example. This package provides several modules:

- Core - contains an Output subclass for use with regression data, and evaluation code for checking model performance using
  standard regression metrics (R^2, explained variance, RMSE, mean absolute error). Also contains simple baseline
  regressions, if your system doesn't outperform these it's not working.
- LibLinear - a wrapper around the LibLinear-java library. This provides linear-SVMs, and other l1 or l2 regularised
  linear regressions.
- LibSVM - a wrapper around the Java version of LibSVM. This provides linear & kernel SVRs, with sigmoid, gaussian and
  polynomial kernels.
- RegressionTrees - a implementation of two types of CART regression trees. The first builds a separate tree per output
  dimension, the second builds a single tree for all outputs.
- SGD - a implementation of stochastic gradient descent for linear regression. It uses the set of gradient optimisers
  in the main Math package, which allow for various forms of regularisation and descent algorithms.
- SLM - a implementation of sparse linear models. It has a co-ordinate descent implementation of ElasticNet, a LARS
  implementation, a LASSO implementation using LARS, and a couple of sequential forward selection algortihms.
- XGBoost - a wrapper around the XGBoost Java API. XGBoost requires a C library accessed via JNI.

One day we might add Gaussian Processes, when it's easy to integrate with a BLAS from Java.

## Clustering

Clustering is the task of grouping input data. The clustering system implemented is single membership, each datapoint is assigned to
one and only one cluster. This package provides two modules:

- Core - contains the Output subclass for use with clustering data, and evaluation code for measuring clustering performance.
- KMeans - an implementation of K-Means using the Java 8 Stream API for parallelisation.

## Anomaly Detection

Anomaly detection is the task of finding outliers or anomalies at prediction time, when trained on a set of non-anomalous data.
This package provides two modules:

- Core - contains the Output subclass for use with anomaly detection data.
- LibSVM - a wrapper around the Java version of LibSVM, which provides a one-class SVM.

## Common

There is a common package which shares code across multiple prediction types. This provides the
base support for LibLinear, LibSVM, nearest neighbour, tree, and XGBoost models.

## Third party models

Tribuo supports loading a number of third party models which were trained outside the system (even in other programming languages)
and scoring them from Java using Tribuo's infrastructure. Currently we support loading ONNX, Tensorflow and XGBoost models.

- ONNX - ONNX (Open Neural Network eXchange) format is used by several deep learning systems as an export format, and there are converters from systems like scikit-learn to the ONNX format.
  Tribuo provides a wrapper around Microsoft's onnx-runtime which can score ONNX models on both CPU and GPU platforms.
- Tensorflow - Tribuo supports loading Tensorflow's classification and regression frozen models and scoring them.
- XGBoost - Tribuo supports loading XGBoost classification and regression models.

## Tensorflow

Tribuo includes experimental support for Tensorflow 1.14. Due to a lack of flexibility in Tensorflow 1.14's Java API, models
still need to be specified in python, and have their graph definitions written out as a protobuf. The Java code accepts this
protobuf and trains a model that can be used purely from Java. It includes a Java serialisation system so all Tensorflow models
can be serialised and deserialised in the same way as other Tribuo models. Tensorflow models run by default on GPU.

This support is experimental while the TF JVM SIG rewrites the Tensorflow Java API.
We participate in the Tensorflow JVM group, and the upcoming releases from that group will include full Java support for training models
without the need to define the model in Python before training.

It includes an example config file, python model generation file and protobuf for an MNIST model, which demonstrates the Tensorflow
interop. In addition to the libraries gathered by the Tribuo tensorflow jar you need to have
libtensorflow\_jni.so and libtensorflow\_framework.so on your java.library.path.

## Java Security Manager

Tribuo uses OLCUT's configuration and provenance systems which use reflection
to construct and inspect classes.  Therefore when running with a Java security
manager you need to give the OLCUT jar appropriate permissions. We have tested
this set of permissions which allows the configuration and provenance systems
to work:

    // OLCUT permissions
    grant codeBase "file:/path/to/olcut/olcut-core-5.1.1.jar" {
            permission java.lang.RuntimePermission "accessDeclaredMembers";
            permission java.lang.reflect.ReflectPermission "suppressAccessChecks";
            permission java.util.logging.LoggingPermission "control";
            permission java.io.FilePermission "<<ALL FILES>>", "read";
            permission java.util.PropertyPermission "*", "read,write";
    };

The read FilePermission can be restricted to the jars which contain configuration 
files, configuration files on disk, and the locations of serialised objects. The 
one here provides access to the complete filesystem, as the necessary read 
locations are program specific and should thus be narrowed based on your 
requirements. If you need to save an OLCUT configuration out then you will also 
need to add write permissions for the save location.

## Release Notes:

- v4.0.0 - Initial public release.
