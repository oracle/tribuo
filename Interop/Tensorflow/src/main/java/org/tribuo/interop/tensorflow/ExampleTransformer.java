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

package org.tribuo.interop.tensorflow;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.math.la.SparseVector;
import org.tensorflow.Tensor;

import java.io.Serializable;
import java.util.List;

/**
 * TensorFlow support is experimental, and may change without a major version bump.
 * <p>
 * Transforms an {@link Example}, extracting the features from it as a {@link Tensor}.
 * <p>
 * This usually densifies the example, so can be a lot larger than the input example.
 */
public interface ExampleTransformer<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Converts an {@link Example} into a {@link Tensor} suitable for supplying as an input to a graph.
     * <p>
     * It generates it as a single example minibatch.
     * @param example The example to convert.
     * @param featureIDMap The id map to convert feature names into id numbers.
     * @return A dense Tensor representing this example.
     */
    public Tensor<?> transform(Example<T> example, ImmutableFeatureMap featureIDMap);

    /**
     * Converts a batch of {@link Example}s into a single {@link Tensor} suitable for supplying as
     * an input to a graph.
     * @param example The examples to convert.
     * @param featureIDMap THe id map to convert feature names into id numbers.
     * @return A dense Tensor representing this minibatch.
     */
    public Tensor<?> transform(List<Example<T>> example, ImmutableFeatureMap featureIDMap);

    /**
     * Converts a {@link SparseVector} representing the features into a {@link Tensor}.
     * <p>
     * It generates it as a single example minibatch.
     * @param vector The features to convert.
     * @return A dense Tensor representing this vector.
     */
    public Tensor<?> transform(SparseVector vector);

    /**
     * Converts a list of {@link SparseVector}s representing a batch of features into a {@link Tensor}.
     * <p>
     * @param vectors The batch of features to convert.
     * @return A dense Tensor representing this minibatch.
     */
    public Tensor<?> transform(List<SparseVector> vectors);

}
