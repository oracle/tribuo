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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.provenance.OutputFactoryProvenance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An interface associated with a specific {@link Output}, which can generate the
 * appropriate Output subclass, and {@link OutputInfo} subclass.
 * <p>
 * Must be {@link Configurable} so it can be loaded from an olcut config file.
 */
public interface OutputFactory<T extends Output<T>> extends Configurable, ProtoSerializable<OutputFactoryProto>, Provenancable<OutputFactoryProvenance>, Serializable {

    /**
     * Parses the {@code V} and generates the appropriate {@link Output} value.
     * <p>
     * Most implementations call toString on the label before parsing it, but this is not required.
     *
     * @param label An input value.
     * @param <V> The type of the input value.
     * @return The parsed Output as an instance of {@code T}.
     */
    public <V> T generateOutput(V label);

    /**
     * Returns the singleton unknown output of type T which can be used for prediction time examples.
     * @return An unknown output.
     */
    public T getUnknownOutput();

    /**
     * Generates the appropriate {@link MutableOutputInfo} so the
     * output values can be tracked by a {@link Dataset} or other
     * aggregate.
     * @return The appropriate subclass of {@link MutableOutputInfo} initialised to zero.
     */
    public MutableOutputInfo<T> generateInfo();

    /**
     * Creates an {@link ImmutableOutputInfo} from the supplied mapping.
     * Requires that the mapping is dense in the integers [0,mapping.size()) and
     * each mapping is unique.
     * <p>
     *     <b>This call is used to import external models, and should not be used for other purposes.</b>
     * </p>
     * @param mapping The mapping to use.
     * @return The appropriate subclass of {@link ImmutableOutputInfo} with a single observation of each element.
     */
    public ImmutableOutputInfo<T> constructInfoForExternalModel(Map<T,Integer> mapping);

    /**
     * Gets an {@link Evaluator} suitable for measuring performance of predictions for the Output subclass.
     * <p>
     * {@link Evaluator} instances are thread safe and immutable, and commonly this is a singleton
     * stored in the {@code OutputFactory} implementation.
     * </p>
     * @return An evaluator.
     */
    public Evaluator<T,? extends Evaluation<T>> getEvaluator();

    /**
     * Gets the output class that this factory supports.
     * @return The output class.
     */
    default public Class<T> getTypeWitness() {
        throw new UnsupportedOperationException("This class must be updated to support protobuf serialization");
    }

    /**
     * Generate a list of outputs from the supplied list of inputs.
     * <p>
     * Makes inputs.size() calls to {@link OutputFactory#generateOutput}.
     * @param inputs The list to convert.
     * @param <V> The type of the inputs
     * @return A list of outputs.
     */
    default public <V> List<T> generateOutputs(List<V> inputs) {
        ArrayList<T> outputs = new ArrayList<>();

        for (V input : inputs) {
            outputs.add(generateOutput(input));
        }

        return outputs;
    }

    /**
     * Deserializes a {@link OutputFactoryProto} into a {@link OutputFactory} subclass.
     * @param proto The proto to deserialize.
     * @return The deserialized OutputFactory.
     */
    public static OutputFactory<?> deserialize(OutputFactoryProto proto) {
        return ProtoUtil.deserialize(proto);
    }

    /**
     * Validates that the mapping can be used as an output info, i.e.
     * that it is dense in the region [0,mapping.size()) - meaning no duplicate
     * ids, each id 0 through mapping.size() is used, and there are no negative ids.
     * @param mapping The mapping to use.
     * @param <T> The type of the output.
     */
    public static <T extends Output<T>> void validateMapping(Map<T,Integer> mapping) {
        Map<Integer,T> reverse = new HashMap<>();
        for (Map.Entry<T,Integer> e : mapping.entrySet()) {
            if (e.getValue() < 0 || e.getValue() >= mapping.size()) {
                throw new IllegalArgumentException("Invalid mapping, expected an integer between 0 and mapping.size(), received " + e.getValue());
            }
            T l = reverse.put(e.getValue(),e.getKey());
            if (l != null) {
                throw new IllegalArgumentException("Invalid mapping, both " + e.getKey() + " and " + l + " map to " + e.getValue());
            }
        }

        if (reverse.size() != mapping.size()) {
            throw new IllegalArgumentException("The Output<->Integer mapping is not a bijection, reverse mapping had " + reverse.size() + " elements, forward mapping had " + mapping.size() + " elements.");
        }
    }
}
