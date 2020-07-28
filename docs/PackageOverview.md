# Package Structure Overview

Tribuo has a modular structure to allow minimal dependencies for any specific
 deployment. We describe the overall package structure below.

## Package Description
The top level project has core modules which define the API, data interactions, 
a math library, and common modules shared across prediction types.

- Core - Provides the main classes and interfaces:
  - Dataset - A list of Examples plus associated feature information, such
   as the number of categories for categorical features, or the mean
    and variance in the case of real-valued features.
  - DataSource - A list of Examples processed from some other format
   and accompanied by the provenance describing the source and processing of
    these Examples. 
  - Example - An array or list of String and value pairs. The Example is typed
   with a subclass of Output that represents the appropriate type of response.
  - Feature - An immutable tuple of String and value. The String is the feature
   name, which is used as the feature's unique identifier.
  - FeatureInfo - A class representing information about the feature, e.g. the 
  number of times it occurred in the dataset.
  - FeatureMap - A map from String to FeatureInfo objects. When immutable, it
   also contains feature id numbers, although these should be treated as an
    implementation detail and not relied upon.
  - Model - A class that can make predictions of a specific Output type.
  - Output - An interface denoting the type of output: regression, 
  multi-label, multi-class, clustering, or anomaly detection.
  - OutputInfo - An interface representing information about the output.
  - Trainer - A class that generates Models based on a Dataset of a specific output type.
  - Prediction - A class that stores the output of a Model when presented
   with an Example for labeling. It contains scores for each of the predicted
    labels. These scores may optionally be a probability distribution.
- Core contains several other packages.
  - dataset - Datasets which provide a view on another dataset, either subsampling it or excluding features below a threshold.
  - datasource - Implementations of DataSource which operate on in memory or simple on disk formats.
  - ensemble - Base classes and interfaces for ensemble models.
  - evaluation - Base evaluation classes for all output types. The package
   also includes evaluation-related classes for cross-validation and train-test 
   splitting.
  - hash - An implementation of feature hashing which obfuscates any feature
   names that are stored in a Model. Hashing prevents feature names from
    leaking out of the training data.
  - provenance - Provenance classes for Tribuo. Provenance tracks the location
   and transformations of datasets, the parameters of trainers, and other
    useful information.
  - sequence - A sequence prediction API.
  - transform - A feature transformation package that can be applied to a full
   dataset or to individual features matched via regexes. It also contains
    wrapping trainers (trainers that wrap another trainer to provide additional
     functionality) and wrapping models to ensure the same transformations are
     always applied at prediction time.
  - util - Utilities for working with arrays, random samples, and other basic operations.
- Data - provides classes which deal with sampled data, columnar data, csv files and text inputs. These classes are mainly optional, and the user is encouraged to provide their own text processing infrastructure.
  - columnar - The columnar package provides many useful base classes for extracting features from columnar data.
  - csv - Builds on the columnar package and supplies infrastructure for working with CSV and other delimiter separated data.
  - sql - Builds on the columnar package and supplies infrastructure for working with JDBC sources.
  - text - An example text processsing infrastructure.
- Math - provides a linear algebra library for working with both sparse
 and dense vectors and matrices.
  - kernel - a set of kernel functions for use in the SGD package (and elsewhere).
  - la - a linear algebra library. It contains functions that are used in the
   SGD implementation. It is not a full BLAS.
  - optimisers - a set of stochastic gradient descent algorithms, including
   SGD + Momentum, AdaGrad, AdaDelta, RMSProp and
  several others. AdaGrad should be considered the default algorithm since
   it works best across the widest range of linear SGD problems.
  - util - various util classes for working with arrays, vectors and matrices.

## Multi-class Classification

Multi-class classification is the act of assigning a single label from a set of labels to a test example.
The classification module has several submodules:

- Core - contains an Output subclass for use with multi-class classification
  tasks, evaluation code for checking model performance, and an
  implementation of Adaboost.SAMME. It also contains simple baseline
   classifiers. If your system doesn't outperform the baseline classifiers, it 
   can be assumed that your system is not working properly.
- DecisionTree - an implementation of CART decision trees.
- Experiments - A set of main functions for training & testing models on any supported dataset. It depends on all the
  classifiers and allows easy comparison between them. This submodule should
   not be imported into other projects, since it is intended purely for
  development and testing.
- Explanations - An implementation of LIME for classification tasks. If you use the columnar data loader, LIME can
  extract more information about the feature domain and provide better explanations.
- LibLinear - a wrapper around the LibLinear-java library. This provides linear-SVMs and other l1 or l2 regularised
  linear classifiers.
- LibSVM - a wrapper around the Java version of LibSVM. This provides linear & kernel SVMs with sigmoid, gaussian and
  polynomial kernels.
- Multinomial Naive Bayes - an implementation of a multinomial naive bayes
 classifier. Since it aims to store a compact in-memory representation of the
  model, it only keeps track of weights for observed feature/class pairs.
- SGD - an implementation of stochastic gradient descent based classifiers. It includes a linear package for logistic
  regression and linear-SVM (using log and hinge losses, respectively), a
   kernel package for training a kernel-SVM using the Pegasos algorithm, and a
    crf package for training a linear-chain CRF. These implementations depend upon the stochastic
  gradient optimisers in the main Math package. The linear and crf packages can use any of the provided gradient optimisers,
  which enforce various different kinds of regularisation or convergence metrics. This is the preferred package for linear
  classification and for sequence classification due to the speed and scalability of the SGD approach.
- XGBoost - a wrapper around the XGBoost Java API. XGBoost requires a C library accessed via JNI. 
  XGBoost is a scalable implementation of gradient boosted trees. It is one of
   the highest accuracy classifiers provided in this project.

## Multi-label Classification

Multi-label classification is the task of predicting a set of labels for a test example rather than just a single label.
This package provides an Output subclass for multi-label
 prediction, evaluation code for checking the performance of a multi-label
  model, and a basic implementation of independent binary predictions. The simplest
multi-label approach is to break each multi-label prediction into n binary predictions, one for each possible label.
To achieve this, the supplied trainer takes another classification trainer and
 uses it to build n models, one per label, which are then run
in sequence on a test example to produce the final multi-label output.

There are many more complicated multi-label approaches in the literature
 that use the label structure to improve
performance. We'll consider adding these based on demand from the community.

## Regression

Regression is the task of predicting real-valued outputs for a test example. This package provides several modules:

- Core - contains an Output subclass for use with regression data, as well as
 evaluation code for checking model performance using standard regression
  metrics (R^2, explained variance, RMSE, and mean absolute error). The
   module also contains simple baseline regressions. If your system fails to
    outperform these baseline models, it can be assumed that your system is not
     working properly.
- LibLinear - a wrapper around the LibLinear-java library. This provides linear-SVMs and other l1 or l2 regularised
  linear regressions.
- LibSVM - a wrapper around the Java version of LibSVM. This provides linear & kernel SVRs with sigmoid, gaussian and
  polynomial kernels.
- RegressionTrees - an implementation of two types of CART regression trees. The
 first type builds a separate tree per output
  dimension, while the second type builds a single tree for all outputs.
- SGD - an implementation of stochastic gradient descent for linear
 regression. It uses the main Math package's set of gradient optimisers, which allow for various regularisation and descent algorithms.
- SLM - an implementation of sparse linear models. It includes a co-ordinate
 descent implementation of ElasticNet, a LARS
  implementation, a LASSO implementation using LARS, and a couple of
   sequential forward selection algorithms.
- XGBoost - a wrapper around the XGBoost Java API. XGBoost requires a C library accessed via JNI.

One day, when it's easy to integrate with a BLAS from Java, we plan to add
 Gaussian Processes.

## Clustering

Clustering is the task of grouping input data. The clustering system
 implemented is single membership -- each datapoint is assigned to
one and only one cluster. This package provides two modules:

- Core - contains the Output subclass for use with clustering data and evaluation code for measuring clustering performance.
- KMeans - an implementation of K-Means using the Java 8 Stream API for parallelisation.

## Anomaly Detection

Anomaly detection is the task of finding outliers or anomalies at prediction
 time using a model trained on non-anomalous data.
This package provides two modules:

- Core - contains the Output subclass for use with anomaly detection data.
- LibSVM - a wrapper around the Java version of LibSVM, which provides a one-class SVM.

## Common

The common package shares code across multiple prediction types. It
 provides the base support for LibLinear, LibSVM, nearest neighbour, tree, and 
 XGBoost models.

## Third party models

Tribuo supports loading a number of third party models which were trained outside the system (even in other programming languages)
and scoring them from Java using Tribuo's infrastructure. Currently, we
 support loading ONNX, TensorFlow and XGBoost models.

- ONNX - [ONNX](https://onnx.ai) (Open Neural Network eXchange) format is used by several deep learning systems as an export format, and there are converters from systems like scikit-learn to the ONNX format.
  Tribuo provides a wrapper around Microsoft's [ONNX Runtime](https://onnxruntime.ai) which can score ONNX models on both CPU and GPU platforms.
- TensorFlow - Tribuo supports loading [TensorFlow](https://tensorflow.org)'s classification and regression frozen models and scoring them.
- XGBoost - Tribuo supports loading [XGBoost](https://xgboost.ai) classification and regression models.

## TensorFlow

Tribuo includes experimental support for TensorFlow 1.14. Due to a lack of flexibility in TensorFlow 1.14's Java API, models
still need to be specified in python, and have their graph definitions written out as a protobuf. The Java code accepts this
protobuf and trains a model that can be used purely from Java. It includes a Java serialisation system so all TensorFlow models
can be serialised and deserialised in the same way as other Tribuo models. TensorFlow models run by default on GPU if one
is available.

This support remains experimental while the TF JVM SIG rewrites the TensorFlow
 Java API.
We participate in the TensorFlow JVM SIG, and the upcoming releases from that group will include full Java support for training models
without the need to define the model in Python before training.

Tribuo demonstrates the TensorFlow interop by including an example config
 file, python model generation file and protobuf for an MNIST model. In
  addition to the libraries gathered by the Tribuo TensorFlow jar, it is
   necessary to include
libtensorflow\_jni.so and libtensorflow\_framework.so on your java.library.path.

