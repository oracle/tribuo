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
import org.tensorflow.types.family.TType;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tensorflow.Tensor;

import java.io.Serializable;
import java.util.List;
import java.util.function.BiFunction;

/**
 * TensorFlow support is experimental, and may change without a major version bump.
 * <p>
 * Converts the {@link Output} into a {@link Tensor} and vice versa.
 * <p>
 * Also provides the loss function for this output type, along with the
 * transformation function for converting the TF graph output into a
 * well formed output float (e.g., a softmax for classification, a sigmoid
 * for multi-label, or the identity function for regression).
 */
public interface OutputTransformer<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * The loss function associated with this prediction type.
     * @return The TF loss function.
     */
    public BiFunction<Ops, Pair<Placeholder<TNumber>,Operand<TNumber>>,Operand<TNumber>> loss();

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
     * Gets the class representing the placeholder type for this input.
     * @param <U> The type.
     * @return The class representing the type.
     */
    public <U extends TType> Class<U> placeholderType();

    /**
     * Converts a {@link Tensor} into a {@link Prediction}.
     * @param tensor The tensor to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @param numValidFeatures The number of valid features used by the prediction.
     * @param example The example to insert into the prediction.
     * @return A prediction object.
     */
    public Prediction<T> transformToPrediction(Tensor tensor, ImmutableOutputInfo<T> outputIDInfo, int numValidFeatures, Example<T> example);

    /**
     * Converts a {@link Tensor} into the specified output type.
     * @param tensor The tensor to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @return A output.
     */
    public T transformToOutput(Tensor tensor, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Converts a {@link Tensor} containing multiple outputs into a list of {@link Prediction}s.
     * @param tensor The tensor to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @param numValidFeatures The number of valid features used by the prediction.
     * @param examples The example to insert into the prediction.
     * @return A list of predictions.
     */
    public List<Prediction<T>> transformToBatchPrediction(Tensor tensor, ImmutableOutputInfo<T> outputIDInfo, int[] numValidFeatures, List<Example<T>> examples);

    /**
     * Converts a {@link Tensor} containing multiple outputs into a list of {@link Output}s.
     * @param tensor The tensor to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @return A list of outputs.
     */
    public List<T> transformToBatchOutput(Tensor tensor, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Converts an {@link Output} into a {@link Tensor} representing it's output.
     * @param output The output to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @return A Tensor representing this output.
     */
    public Tensor transform(T output, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Converts a list of {@link Example} into a {@link Tensor} representing all the outputs
     * in the list. It accepts a list of Example rather than a list of Output for efficiency reasons.
     * @param examples The examples to convert.
     * @param outputIDInfo The output info to use to identify the outputs.
     * @return A Tensor representing all the supplied Outputs.
     */
    public Tensor transform(List<Example<T>> examples, ImmutableOutputInfo<T> outputIDInfo);

    /**
     * Does this OutputTransformer generate probabilities.
     * @return True if it produces a probability distribution in the {@link Prediction}.
     */
    public boolean generatesProbabilities();

}
