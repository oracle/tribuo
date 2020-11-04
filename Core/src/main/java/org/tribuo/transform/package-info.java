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
 * Provides infrastructure for applying transformations to a {@link org.tribuo.Dataset}.
 * <p>
 * This package is the necessary infrastructure for transformations. The workflow is first to build a
 * {@link org.tribuo.transform.TransformationMap} which represents the
 * {@link org.tribuo.transform.Transformation}s and the order that they should be applied to each
 * {@link org.tribuo.Feature}. This can be applied to a Dataset to produce a
 * {@link org.tribuo.transform.TransformerMap} which contains a fitted set of
 * {@link org.tribuo.transform.Transformer}s which can be used to apply the transformation to any
 * other Dataset (e.g., to apply the same transformation to training and test sets), or to be used at prediction
 * time to stream data through.
 * <p>
 * It also provides a {@link org.tribuo.transform.TransformTrainer} which accepts a
 * TransformationMap and an inner {@link org.tribuo.Trainer} and produces a
 * {@link org.tribuo.transform.TransformedModel} which automatically transforms it's input data at
 * prediction time.
 */
package org.tribuo.transform;