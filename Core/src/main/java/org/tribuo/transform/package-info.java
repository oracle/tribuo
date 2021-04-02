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
 * {@link org.tribuo.transform.Transformation}s and the order that they should be applied to the specified
 * {@link org.tribuo.Feature}s. This can be applied to a Dataset to produce a
 * {@link org.tribuo.transform.TransformerMap} which contains a fitted set of
 * {@link org.tribuo.transform.Transformer}s which can be used to apply the transformation to any
 * other Dataset (e.g., to apply the same transformation to training and test sets), or to be used at prediction
 * time to stream data through.
 * <p>
 * It also provides a {@link org.tribuo.transform.TransformTrainer} which accepts a
 * TransformationMap and an inner {@link org.tribuo.Trainer} and produces a
 * {@link org.tribuo.transform.TransformedModel} which automatically transforms it's input data at
 * prediction time.
 * <p>
 * Transformations don't produce new {@link org.tribuo.Feature}s - they only modify the values of existing ones.
 * When doing so they can be instructed to treat Features that are absent due to sparsity as zero or as
 * not existing at all. Independently, we can explicitly add zero-valued Features by densifying the dataset
 * before the transformation is fit or before it is applied. Once they exist these Features can be altered by
 * {@link org.tribuo.transform.Transformer}s and are visible to {@link org.tribuo.transform.Transformation}s which are
 * being fit.
 * <p>
 * The transformation fitting methods have two parameters which alter their behaviour: {@code includeImplicitZeroFeatures}
 * and {@code densify}. {@code includeImplicitZeroFeatures} controls if the transformation incorporates the implicit zero
 * valued features (i.e., the ones not present in the example but are present in the dataset's
 * {@link org.tribuo.FeatureMap}) when building the transformation statistics. This is
 * important when working with, e.g. {@link org.tribuo.transform.transformations.IDFTransformation} as it allows correct
 * computation of the inverse document frequency, but can be detrimental to features which are one-hot encodings of
 * categoricals (as they have many more implicit zeros). {@code densify} controls if the example or dataset should have
 * its implicit zero valued features converted into explicit zero valued features (i.e., it makes a sparse example into
 * a dense one which contains an explicit value for every feature known to the dataset) before the transformation is
 * applied, and transformations are only applied to feature values which are present.
 * <p>
 * These parameters interact to form 4 possibilities:
 * <ul>
 *     <li>Both false: transformations are only fit on explicit feature values, and only applied to explicit feature values</li>
 *     <li>Both true: transformations include explicit features and implicit zeros, and implicit zeros are converted into explicit zeros and transformed</li>
 *     <li>{@code includeImplicitZeroFeatures} is true, {@code densify} is false: the implicit zeroes are used to fit
 *     the transformation, but not modified when the transformation is applied. This is most useful when working with
 *     text data where you want to compute IDF style statistics</li>
 *     <li>{@code includeImplicitZeroFeatures} is false, {@code densify} is true: the implicit zeros are not used to
 *     fit the transformation, but are converted to explicit zeros and transformed. This is less useful than the other
 *     three combinations, but could be used to move the minimum value, or when zero is not appropriate for a missing
 *     value and needs to be transformed.</li>
 * </ul>
 * One further option is to call {@link org.tribuo.MutableDataset#densify} before passing the data to
 * {@link org.tribuo.transform.TransformTrainer#train}, which is equivalent to setting {@code includeImplicitZeroFeatures}
 * to true and {@code densify} to true. To sum up, in the context of transformations {@code includeImplicitZeroFeatures}
 * determines whether (implicit) zero-values features are <i>measured</i> and {@code densify} determines whether
 * they can be <i>altered</i>. 
 */
package org.tribuo.transform;
