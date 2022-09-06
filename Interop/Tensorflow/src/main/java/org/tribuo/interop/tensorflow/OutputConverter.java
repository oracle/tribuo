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
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tensorflow.Operand;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.types.family.TNumber;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tensorflow.Tensor;
import org.tribuo.interop.tensorflow.protos.OutputConverterProto;
import org.tribuo.protos.ProtoSerializable;

import java.io.Serializable;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Converts the {@link Output} into a {@link Tensor} and vice versa.
 * <p>
 * Also provides the loss function for this output type, along with the
 * function which converts the TF graph output into a
 * well formed output float (e.g., a softmax for classification, a sigmoid
 * for multi-label, or the identity function for regression).
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 * @param <T> The output type.
 */
public interface OutputConverter<T extends Output<T>> extends Configurable, ProtoSerializable<OutputConverterProto>, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * The loss function associated with this prediction type.
     * @return The TF loss function.
     */
    public BiFunction<Ops, Pair<Placeholder<? extends TNumber>,Operand<TNumber>>,Operand<TNumber>> loss();

    /**
     * Produces an output transformation function that applies the operation to
     * the graph from the supplied {@link Ops}, taking a graph output operation.
     * <p>
     * For example this function will apply a softmax for classification, a sigmoid
     * for multi-label, or the identity function for regression.
     * @param <U> The type of the graph output.
     * @return A function which applies the appropriate transformation function.
     */
    public <U extends TNumber> BiFunction<Ops, Operand<U>, Op> outputTransformFunction();

    /**
     * Converts a {@link Tensor} into a {@link Prediction}.
     * @param tensor The tensor to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @param numValidFeatures The number of valid features used by the prediction.
     * @param example The example to insert into the prediction.
     * @return A prediction object.
     */
    public Prediction<T> convertToPrediction(Tensor tensor, ImmutableOutputInfo<T> outputIDInfo, int numValidFeatures, Example<T> example);

    /**
     * Converts a {@link Tensor} into the specified output type.
     * @param tensor The tensor to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @return A output.
     */
    public T convertToOutput(Tensor tensor, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Converts a {@link Tensor} containing multiple outputs into a list of {@link Prediction}s.
     * @param tensor The tensor to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @param numValidFeatures The number of valid features used by the prediction.
     * @param examples The example to insert into the prediction.
     * @return A list of predictions.
     */
    public List<Prediction<T>> convertToBatchPrediction(Tensor tensor, ImmutableOutputInfo<T> outputIDInfo, int[] numValidFeatures, List<Example<T>> examples);

    /**
     * Converts a {@link Tensor} containing multiple outputs into a list of {@link Output}s.
     * @param tensor The tensor to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @return A list of outputs.
     */
    public List<T> convertToBatchOutput(Tensor tensor, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Converts an {@link Output} into a {@link Tensor} representing it's output.
     * @param output The output to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @return A Tensor representing this output.
     */
    public Tensor convertToTensor(T output, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Converts a list of {@link Example} into a {@link Tensor} representing all the outputs
     * in the list. It accepts a list of Example rather than a list of Output for efficiency reasons.
     * @param examples The examples to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @return A Tensor representing all the supplied Outputs.
     */
    public Tensor convertToTensor(List<Example<T>> examples, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Does this OutputConverter generate probabilities.
     * @return True if it produces a probability distribution in the {@link Prediction}.
     */
    public boolean generatesProbabilities();

    /**
     * The type witness used when deserializing the TensorFlow model from a protobuf.
     * <p>
     * The default implementation throws {@link UnsupportedOperationException} for compatibility with implementations
     * which don't use protobuf serialization. This implementation will be removed in the next major version of
     * Tribuo.
     * @return The output class this object produces.
     */
    default public Class<T> getTypeWitness() {
        throw new UnsupportedOperationException("This implementation should be replaced to support protobuf serialization");
    }
}
