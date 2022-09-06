/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.interop.tensorflow.protos.FeatureConverterProto;
import org.tribuo.math.la.SGDVector;
import org.tribuo.protos.ProtoSerializable;

import java.io.Serializable;
import java.util.List;
import java.util.Set;

/**
 * Transforms an {@link Example} or {@link SGDVector}, extracting the features from it as a {@link TensorMap}.
 * <p>
 * This usually densifies the example, so can be a lot larger than the input example.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 */
public interface FeatureConverter extends Configurable, ProtoSerializable<FeatureConverterProto>, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Converts an {@link Example} into a {@link TensorMap} suitable for supplying as an input to a graph.
     * <p>
     * It generates it as a single example minibatch.
     * @param example The example to convert.
     * @param featureIDMap The id map to convert feature names into id numbers.
     * @return A TensorMap (similar to a TF Python feed_dict) representing the features in this example.
     */
    public TensorMap convert(Example<?> example, ImmutableFeatureMap featureIDMap);

    /**
     * Converts a batch of {@link Example}s into a single {@link TensorMap} suitable for supplying as
     * an input to a graph.
     * @param example The examples to convert.
     * @param featureIDMap THe id map to convert feature names into id numbers.
     * @return A TensorMap (similar to a TF Python feed_dict) representing the features in this minibatch.
     */
    public TensorMap convert(List<? extends Example<?>> example, ImmutableFeatureMap featureIDMap);

    /**
     * Converts a {@link SGDVector} representing the features into a {@link TensorMap}.
     * <p>
     * It generates it as a single example minibatch.
     * @param vector The features to convert.
     * @return A TensorMap (similar to a TF Python feed_dict) representing this vector.
     */
    public TensorMap convert(SGDVector vector);

    /**
     * Converts a list of {@link SGDVector}s representing a batch of features into a {@link TensorMap}.
     * @param vectors The batch of features to convert.
     * @return A TensorMap (similar to a TF Python feed_dict) representing this minibatch.
     */
    public TensorMap convert(List<? extends SGDVector> vectors);

    /**
     * Gets a view of the names of the inputs this converter produces.
     * @return The input names.
     */
    public Set<String> inputNamesSet();

}
