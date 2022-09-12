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

package org.tribuo.ensemble;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.EnsembleCombinerProto;
import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXRef;

import java.io.Serializable;
import java.util.List;
import java.util.logging.Logger;

/**
 * An interface for combining predictions. Implementations should be final and immutable.
 */
public interface EnsembleCombiner<T extends Output<T>> extends Configurable, ProtoSerializable<EnsembleCombinerProto>, Provenancable<ConfiguredObjectProvenance>, Serializable {

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
     * Exports this ensemble combiner into the ONNX context of its input.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     * <p>
     * For compatibility reasons this method has a default implementation, though
     * when called it will throw an {@code IllegalStateException}. In a future
     * version this method will not have a default implementation and ensemble combiners
     * will be required to provide ONNX support.
     * @param input the node to be ensembled according to this implementation.
     * @return The leaf node of the graph of operations added to ensemble input.
     */
    default ONNXNode exportCombiner(ONNXNode input) {
        Logger.getLogger(this.getClass().getName()).severe("Tried to export an ensemble combiner to ONNX format, but this is not implemented.");
        throw new IllegalStateException("This ensemble cannot be exported as the combiner '" + this.getClass() + "' uses the default implementation of EnsembleCombiner.exportCombiner.");
    }

    /**
     * Exports this ensemble combiner into the ONNX context of its input.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     * <p>
     * For compatibility reasons this method has a default implementation, though
     * when called it will throw an {@code IllegalStateException}. In a future
     * version this method will not have a default implementation and ensemble combiners
     * will be required to provide ONNX support.
     * @param input the node to be ensembled according to this implementation.
     * @param weight The node of weights for ensembling.
     * @param <U> The type of the weights input reference.
     * @return The leaf node of the graph of operations added to ensemble input.
     */
    default <U extends ONNXRef<?>> ONNXNode exportCombiner(ONNXNode input, U weight) {
        Logger.getLogger(this.getClass().getName()).severe("Tried to export an ensemble combiner to ONNX format, but this is not implemented.");
        throw new IllegalStateException("This ensemble cannot be exported as the combiner '" + this.getClass() + "' uses the default implementation of EnsembleCombiner.exportCombiner.");
    }

    /**
     * The type witness used when deserializing the combiner from a protobuf.
     * <p>
     * The default implementation throws {@link UnsupportedOperationException} for compatibility with implementations
     * which don't use protobuf serialization. This implementation will be removed in the next major version of
     * Tribuo.
     * @return The output class this object produces.
     */
    default public Class<T> getTypeWitness() {
        throw new UnsupportedOperationException("This implementation needs to be updated to support protobuf serialization");
    }

    /**
     * Deserialization helper for EnsembleCombiner.
     * @param proto The proto to deserialize.
     * @return The combiner.
     */
    public static EnsembleCombiner<?> deserialize(EnsembleCombinerProto proto) {
        return ProtoUtil.deserialize(proto);
    }
}
