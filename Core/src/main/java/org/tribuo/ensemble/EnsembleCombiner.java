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

package org.tribuo.ensemble;

import ai.onnx.proto.OnnxMl;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.onnx.ONNXContext;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * An interface for combining predictions. Implementations should be final and immutable.
 */
public interface EnsembleCombiner<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Combine the predictions.
     * @param outputInfo The output domain.
     * @param predictions The predictions to combine.
     * @return The ensemble prediction.
     */
    public Prediction<T> combine(ImmutableOutputInfo<T> outputInfo, List<Prediction<T>> predictions);

    /**
     * Combine the supplied predictions. predictions.size() must equal weights.length.
     * @param outputInfo The output domain.
     * @param predictions The predictions to combine.
     * @param weights The weights to use for each prediction.
     * @return The ensemble prediction.
     */
    public Prediction<T> combine(ImmutableOutputInfo<T> outputInfo, List<Prediction<T>> predictions, float[] weights);

    /**
     * Exports this ensemble combiner as a non-empty list of ONNX NodeProtos.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     * <p>
     * For compatibility reasons the default implementation returns an empty list, and
     * combiners which use the default implementation will not be able to export
     * their models to ONNX. It is recommended that this method is overridden to
     * support ONNX export, and in a future version of Tribuo this default implementation
     * will be removed.
     * @param context The ONNX context object for name generation.
     * @param input The name of the input tensor to combine.
     * @param output The name of the combined output.
     * @return A list of node protos representing the combiner operation.
     */
    default public List<OnnxMl.NodeProto> exportCombiner(ONNXContext context, String input, String output) {
        Logger.getLogger(this.getClass().getName()).severe("Tried to export an ensemble combiner to ONNX format, but this is not implemented.");
        return Collections.emptyList();
    }

    /**
     * Exports this ensemble combiner as a non-empty list of ONNX NodeProtos.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     * <p>
     * For compatibility reasons the default implementation returns an empty list, and
     * combiners which use the default implementation will not be able to export
     * their models to ONNX. It is recommended that this method is overridden to
     * support ONNX export, and in a future version of Tribuo this default implementation
     * will be removed.
     * @param context The ONNX context object for name generation.
     * @param input The name of the input tensor to combine.
     * @param output The name of the combined output.
     * @param weight The name of the combination weight initializer.
     * @return A list of node protos representing the combiner operation.
     */
    default public List<OnnxMl.NodeProto> exportCombiner(ONNXContext context, String input, String output, String weight) {
        Logger.getLogger(this.getClass().getName()).severe("Tried to export an ensemble combiner to ONNX format, but this is not implemented.");
        return Collections.emptyList();
    }

}
