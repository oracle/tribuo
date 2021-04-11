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

package org.tribuo.interop.tensorflow.sequence;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.interop.tensorflow.TensorMap;
import org.tribuo.sequence.SequenceExample;

import java.io.Serializable;
import java.util.List;

/**
 * Converts a sequence example into a feed dict suitable for Tensorflow.
 */
public interface SequenceExampleTransformer<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Encodes an example as a feed dict.
     *
     * @param example the input example
     * @param featureMap feature domain
     * @return a map from graph placeholder names to their fed-in values.
     */
    TensorMap encode(SequenceExample<T> example, ImmutableFeatureMap featureMap);

    /**
     * Encodes a batch of examples as a feed dict.
     *
     * @param batch a batch of examples.
     * @param featureMap feature domain
     * @return a map from graph placeholder names to their fed-in values.
     */
    TensorMap encode(List<SequenceExample<T>> batch, ImmutableFeatureMap featureMap);

}
