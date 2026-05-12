#!/usr/bin/env bash

# Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script rebuilds all of Tribuo's protobufs. It expects OLCUT to be checked out in a sibling directory to Tribuo.
# It does not rebuild the ONNX protobuf in Util/ONNXExport.

if [[ $PROTOC_PATH != "" ]] ; then
  PROTOC=$PROTOC_PATH;
else
  PROTOC="protoc";
fi

PROTO_DEPS="
--proto_path=../olcut/olcut-config-protobuf/src/main/resources/
--proto_path=Core/src/main/resources/protos
--proto_path=Math/src/main/resources/protos
--proto_path=Common/LibLinear/src/main/resources/protos
--proto_path=Common/LibSVM/src/main/resources/protos
--proto_path=Common/NearestNeighbour/src/main/resources/protos
--proto_path=Common/SGD/src/main/resources/protos
--proto_path=Common/Trees/src/main/resources/protos
--proto_path=Common/XGBoost/src/main/resources/protos
"

# Core
$PROTOC $PROTO_DEPS --java_out=Core/src/main/java Core/src/main/resources/protos/tribuo-core.proto Core/src/main/resources/protos/tribuo-core-impl.proto
$PROTOC $PROTO_DEPS --proto_path=Core/src/test/resources/protos --java_out=Core/src/test/java Core/src/test/resources/protos/tribuo-core-test.proto

# Math
$PROTOC $PROTO_DEPS --java_out=Math/src/main/java Math/src/main/resources/protos/tribuo-math.proto Math/src/main/resources/protos/tribuo-math-impl.proto

# Common
$PROTOC $PROTO_DEPS --java_out=Common/LibLinear/src/main/java Common/LibLinear/src/main/resources/protos/tribuo-liblinear.proto
$PROTOC $PROTO_DEPS --java_out=Common/LibSVM/src/main/java Common/LibSVM/src/main/resources/protos/tribuo-libsvm.proto
$PROTOC $PROTO_DEPS --java_out=Common/NearestNeighbour/src/main/java Common/NearestNeighbour/src/main/resources/protos/tribuo-common-knn.proto
$PROTOC $PROTO_DEPS --java_out=Common/SGD/src/main/java Common/SGD/src/main/resources/protos/tribuo-sgd.proto
$PROTOC $PROTO_DEPS --java_out=Common/Trees/src/main/java Common/Trees/src/main/resources/protos/tribuo-tree.proto
$PROTOC $PROTO_DEPS --java_out=Common/XGBoost/src/main/java Common/XGBoost/src/main/resources/protos/tribuo-xgboost.proto

# Interop
$PROTOC $PROTO_DEPS --proto_path=Interop/OCI/src/main/resources/protos --java_out=Interop/OCI/src/main/java Interop/OCI/src/main/resources/protos/tribuo-oci.proto
$PROTOC $PROTO_DEPS --proto_path=Interop/ONNX/src/main/resources/protos --java_out=Interop/ONNX/src/main/java Interop/ONNX/src/main/resources/protos/tribuo-onnx.proto
$PROTOC $PROTO_DEPS --proto_path=Interop/Tensorflow/src/main/resources/protos --java_out=Interop/Tensorflow/src/main/java Interop/Tensorflow/src/main/resources/protos/tribuo-tensorflow.proto

# AnomalyDetection
$PROTOC $PROTO_DEPS --proto_path=AnomalyDetection/Core/src/main/resources/protos --java_out=AnomalyDetection/Core/src/main/java AnomalyDetection/Core/src/main/resources/protos/tribuo-anomaly-core.proto
$PROTOC $PROTO_DEPS --proto_path=AnomalyDetection/LibSVM/src/main/resources/protos --java_out=AnomalyDetection/LibSVM/src/main/java AnomalyDetection/LibSVM/src/main/resources/protos/tribuo-anomaly-libsvm.proto

# Classification
$PROTOC $PROTO_DEPS --proto_path=Classification/Core/src/main/resources/protos --java_out=Classification/Core/src/main/java Classification/Core/src/main/resources/protos/tribuo-classification-core.proto
$PROTOC $PROTO_DEPS --proto_path=Classification/LibSVM/src/main/resources/protos --java_out=Classification/LibSVM/src/main/java Classification/LibSVM/src/main/resources/protos/tribuo-classification-libsvm.proto
$PROTOC $PROTO_DEPS --proto_path=Classification/MultinomialNaiveBayes/src/main/resources/protos --java_out=Classification/MultinomialNaiveBayes/src/main/java Classification/MultinomialNaiveBayes/src/main/resources/protos/tribuo-classification-mnb.proto
$PROTOC $PROTO_DEPS --proto_path=Classification/SGD/src/main/resources/protos --java_out=Classification/SGD/src/main/java Classification/SGD/src/main/resources/protos/tribuo-classification-sgd.proto

# Clustering
$PROTOC $PROTO_DEPS --proto_path=Clustering/Core/src/main/resources/protos --java_out=Clustering/Core/src/main/java Clustering/Core/src/main/resources/protos/tribuo-clustering-core.proto
$PROTOC $PROTO_DEPS --proto_path=Clustering/GMM/src/main/resources/protos --java_out=Clustering/GMM/src/main/java Clustering/GMM/src/main/resources/protos/tribuo-clustering-gmm.proto
$PROTOC $PROTO_DEPS --proto_path=Clustering/Hdbscan/src/main/resources/protos --java_out=Clustering/Hdbscan/src/main/java Clustering/Hdbscan/src/main/resources/protos/tribuo-clustering-hdbscan.proto
$PROTOC $PROTO_DEPS --proto_path=Clustering/KMeans/src/main/resources/protos --java_out=Clustering/KMeans/src/main/java Clustering/KMeans/src/main/resources/protos/tribuo-clustering-kmeans.proto

# MultiLabel
$PROTOC $PROTO_DEPS --proto_path=MultiLabel/Core/src/main/resources/protos --java_out=MultiLabel/Core/src/main/java MultiLabel/Core/src/main/resources/protos/tribuo-multilabel-core.proto
$PROTOC $PROTO_DEPS --proto_path=MultiLabel/SGD/src/main/resources/protos --java_out=MultiLabel/SGD/src/main/java MultiLabel/SGD/src/main/resources/protos/tribuo-multilabel-sgd.proto

# Regression
$PROTOC $PROTO_DEPS --proto_path=Regression/Core/src/main/resources/protos --java_out=Regression/Core/src/main/java Regression/Core/src/main/resources/protos/tribuo-regression-core.proto
$PROTOC $PROTO_DEPS --proto_path=Regression/GaussianProcess/src/main/resources/protos --java_out=Regression/GaussianProcess/src/main/java Regression/GaussianProcess/src/main/resources/protos/tribuo-regression-gp.proto
$PROTOC $PROTO_DEPS --proto_path=Regression/LibSVM/src/main/resources/protos --java_out=Regression/LibSVM/src/main/java Regression/LibSVM/src/main/resources/protos/tribuo-regression-libsvm.proto
$PROTOC $PROTO_DEPS --proto_path=Regression/RegressionTree/src/main/resources/protos --java_out=Regression/RegressionTree/src/main/java Regression/RegressionTree/src/main/resources/protos/tribuo-regression-tree.proto
$PROTOC $PROTO_DEPS --proto_path=Regression/SGD/src/main/resources/protos --java_out=Regression/SGD/src/main/java Regression/SGD/src/main/resources/protos/tribuo-regression-sgd.proto
$PROTOC $PROTO_DEPS --proto_path=Regression/SLM/src/main/resources/protos --java_out=Regression/SLM/src/main/java Regression/SLM/src/main/resources/protos/tribuo-regression-slm.proto
