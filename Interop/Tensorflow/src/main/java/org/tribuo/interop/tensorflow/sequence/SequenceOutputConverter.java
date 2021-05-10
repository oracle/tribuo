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
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.interop.tensorflow.TensorMap;
import org.tribuo.sequence.SequenceExample;
import org.tensorflow.Tensor;

import java.io.Serializable;
import java.util.List;

/**
 * Converts a TensorFlow output tensor into a list of predictions, and a Tribuo sequence example into
 * a Tensorflow tensor suitable for training.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 */
public interface SequenceOutputConverter<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Decode a tensor of graph output into a list of predictions for the input sequence.
     *
     * @param output graph output
     * @param input original input sequence example
     * @param labelMap label domain
     * @return the model's decoded prediction for the input sequence.
     */
    List<Prediction<T>> decode(Tensor output, SequenceExample<T> input, ImmutableOutputInfo<T> labelMap);

    /**
     * Decode graph output tensors corresponding to a batch of input sequences.
     *
     * @param outputs a tensor corresponding to a batch of outputs.
     * @param inputBatch the original input batch.
     * @param labelMap label domain
     * @return the model's decoded predictions, one for each example in the input batch.
     */
    List<List<Prediction<T>>> decode(Tensor outputs, List<SequenceExample<T>> inputBatch, ImmutableOutputInfo<T> labelMap);

    /**
     * Encodes an example's label as a feed dict.
     *
     * @param example the input example
     * @param labelMap label domain
     * @return a map from graph placeholder names to their fed-in values.
     */
    TensorMap encode(SequenceExample<T> example, ImmutableOutputInfo<T> labelMap);

    /**
     * Encodes a batch of labels as a feed dict.
     *
     * @param batch a batch of examples.
     * @param labelMap label domain
     * @return a map from graph placeholder names to their fed-in values.
     */
    TensorMap encode(List<SequenceExample<T>> batch, ImmutableOutputInfo<T> labelMap);

}