/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Provides an interface to TensorFlow, allowing the training of non-sequential models using any supported
 * Tribuo output type. Sequential model support is found in {@link org.tribuo.interop.tensorflow.sequence}.
 * <p>
 * Tribuo's TensorFlow support operates in Graph mode, as in v0.3.1 that is the only way to access
 * gradients. The set of supported gradients is determined by TensorFlow, and not all gradients are
 * available in TensorFlow Java in v0.3.1. Unsupported gradients will trigger an exception when the
 * train method is called.
 * <p>
 * Models can store their trained parameters in two ways, either inside the Tribuo serialized model file
 * (using {@link org.tribuo.interop.tensorflow.TensorFlowTrainer.TFModelFormat#TRIBUO_NATIVE}) or as a
 * TensorFlow checkpoint folder on disk (using {@link org.tribuo.interop.tensorflow.TensorFlowTrainer.TFModelFormat#CHECKPOINT}).
 * The choice is made at training time, as they result in slightly different TF graph structures.
 * <p>
 * Similarly there are two supported kinds of {@link org.tribuo.interop.ExternalModel} for TensorFlow,
 * {@link org.tribuo.interop.tensorflow.TensorFlowSavedModelExternalModel} which loads a {@code SavedModelBundle}
 * and always reads from that path, and {@link org.tribuo.interop.tensorflow.TensorFlowFrozenExternalModel} which
 * loads a TensorFlow v1 frozen graph and stores the graph inside the Tribuo serialized object.
 * <p>
 * There are two main interfaces for interacting with TensorFlow Tensors,
 * {@link org.tribuo.interop.tensorflow.FeatureConverter} and
 * {@link org.tribuo.interop.tensorflow.OutputConverter}, with
 * provide conversions to and from Tribuo's features and outputs respectively. There
 * are implementations of a dense feature transformation and one for images as 3d arrays,
 * along with output converters for {@link org.tribuo.classification.Label}
 * {@link org.tribuo.regression.Regressor}, and {@link org.tribuo.multilabel.MultiLabel}.
 * The loss function and output transformation used is controlled by the {@link org.tribuo.interop.tensorflow.OutputConverter},
 * if a different one is desired then users are recommended to implement that interface separately.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 */
package org.tribuo.interop.tensorflow;