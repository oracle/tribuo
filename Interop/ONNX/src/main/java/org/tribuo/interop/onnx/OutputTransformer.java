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

package org.tribuo.interop.onnx;

import ai.onnxruntime.OnnxValue;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;

import java.io.Serializable;
import java.util.List;

/**
 * Converts an {@link OnnxValue} into an {@link Output} or a {@link Prediction}.
 * <p>
 * N.B. ONNX support is experimental, and may change without a major version bump.
 */
public interface OutputTransformer<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Converts a {@link OnnxValue} into a {@link Prediction}.
     * @param value The value to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @param numValidFeatures The number of valid features used by the prediction.
     * @param example The example to insert into the prediction.
     * @return A prediction object.
     */
    public Prediction<T> transformToPrediction(List<OnnxValue> value, ImmutableOutputInfo<T> outputIDInfo, int numValidFeatures, Example<T> example);

    /**
     * Converts a {@link OnnxValue} into the specified output type.
     * @param value The value to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @return A output.
     */
    public T transformToOutput(List<OnnxValue> value, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Converts a {@link OnnxValue} containing multiple outputs into a list of {@link Prediction}s.
     * @param value The value to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @param numValidFeatures The number of valid features used by the prediction.
     * @param examples The example to insert into the prediction.
     * @return A list of predictions.
     */
    public List<Prediction<T>> transformToBatchPrediction(List<OnnxValue> value, ImmutableOutputInfo<T> outputIDInfo, int[] numValidFeatures, List<Example<T>> examples);

    /**
     * Converts a {@link OnnxValue} containing multiple outputs into a list of {@link Output}s.
     * @param value The value to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @return A list of outputs.
     */
    public List<T> transformToBatchOutput(List<OnnxValue> value, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Does this OutputTransformer generate probabilities.
     * @return True if it produces a probability distribution in the {@link Prediction}.
     */
    public boolean generatesProbabilities();

}
