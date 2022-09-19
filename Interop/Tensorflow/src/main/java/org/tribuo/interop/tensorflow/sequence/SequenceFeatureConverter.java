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

package org.tribuo.interop.tensorflow.sequence;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.interop.tensorflow.TensorMap;
import org.tribuo.interop.tensorflow.protos.SequenceFeatureConverterProto;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.sequence.SequenceExample;

import java.io.Serializable;
import java.util.List;
import java.util.Set;

/**
 * Converts a sequence example into a feed dict suitable for TensorFlow.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 */
public interface SequenceFeatureConverter extends Configurable, ProtoSerializable<SequenceFeatureConverterProto>, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Encodes an example as a feed dict.
     *
     * @param example the input example
     * @param featureMap feature domain
     * @return a map from graph placeholder names to their fed-in values.
     */
    TensorMap encode(SequenceExample<?> example, ImmutableFeatureMap featureMap);

    /**
     * Encodes a batch of examples as a feed dict.
     *
     * @param batch a batch of examples.
     * @param featureMap feature domain
     * @return a map from graph placeholder names to their fed-in values.
     */
    TensorMap encode(List<? extends SequenceExample<?>> batch, ImmutableFeatureMap featureMap);

    /**
     * Gets a view of the names of the inputs this converter produces.
     * @return The input names.
     */
    Set<String> inputNamesSet();
}
