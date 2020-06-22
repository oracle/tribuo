/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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
 * Provides an interface to Tensorflow, allowing the training of non-sequential models using any supported
 * Tribuo output type.
 * <p>
 * There are two main interfaces for interacting with Tensorflow Tensors,
 * {@link org.tribuo.interop.tensorflow.ExampleTransformer} and
 * {@link org.tribuo.interop.tensorflow.OutputTransformer}, with
 * provide conversions to and from Tribuo's features and outputs respectively. There
 * are implementations of a dense feature transformation and one for images as 3d arrays,
 * along with output transformers for {@link org.tribuo.classification.Label}
 * and {@link org.tribuo.regression.Regressor}.
 * <p>
 * N.B. Tensorflow support is experimental and may change without a major version bump.
 */
package org.tribuo.interop.tensorflow;