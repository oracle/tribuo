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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoSerializableMapField;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.OutputProto;
import org.tribuo.protos.core.PredictionImplProto;
import org.tribuo.protos.core.PredictionProto;

/**
 * A prediction made by a {@link Model}.
 * Contains the output, and optionally and a map of scores over the possible outputs.
 * If hasProbabilities() == true then it has a probability
 * distribution over outputs otherwise it is unnormalized scores over outputs.
 * <p>
 * If possible it also contains the number of features that were used to make a prediction,
 * and how many features originally existed in the {@link Example}.
 */
@ProtoSerializableClass(version = Prediction.CURRENT_VERSION, serializedDataClass = PredictionImplProto.class)
public class Prediction<T extends Output<T>> implements ProtoSerializable<PredictionProto>, Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * The current protobuf serialization version of this class.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * The example which was used to generate this prediction.
     */
    @ProtoSerializableField
    private final Example<T> example;

    /**
     * The output assigned by a model.
     */
    @ProtoSerializableField
    private final T output;

    /**
     * Does outputScores contain probabilities or scores?
     */
    @ProtoSerializableField
    private final boolean probability;

    /**
     * How many features were used by the model.
     */
    @ProtoSerializableField
    private final int numUsed;

    /**
     * How many features were set in the example.
     */
    @ProtoSerializableField
    private final int exampleSize;

    /**
     * A map from output name to output object, which contains the score.
     */
    @ProtoSerializableMapField()
    private final Map<String,T> outputScores;

    /**
     * Constructs a prediction from the supplied arguments.
     * @param output The predicted output (i.e., the one with the maximum score).
     * @param outputScores The output score distribution.
     * @param numUsed The number of features used to make the prediction.
     * @param exampleSize The size of the input example.
     * @param example The input example.
     * @param probability Are the scores probabilities?
     */
    private Prediction(T output, Map<String,T> outputScores, int numUsed, int exampleSize, Example<T> example, boolean probability) {
        this.example = example;
        this.outputScores = outputScores;
        this.numUsed = numUsed;
        this.exampleSize = exampleSize;
        this.output = output;
        this.probability = probability;
    }

    /**
     * Constructs a prediction from the supplied arguments.
     * @param output The predicted output (i.e., the one with the maximum score).
     * @param outputScores The output score distribution.
     * @param numUsed The number of features used to make the prediction.
     * @param example The input example.
     * @param probability Are the scores probabilities?
     */
    public Prediction(T output, Map<String,T> outputScores, int numUsed, Example<T> example, boolean probability) {
        this(output,outputScores,numUsed,example.size(),example,probability);
    }

    /**
     * Constructs a prediction from the supplied arguments.
     * @param output The predicted output.
     * @param numUsed The number of features used to make the prediction.
     * @param example The input example.
     */
    public Prediction(T output, int numUsed, Example<T> example) {
        this(output,Collections.emptyMap(),numUsed,example.size(),example,false);
    }

    /**
     * Constructs a prediction from the supplied arguments.
     * @param other The prediction to copy.
     * @param numUsed The number of features used to make the prediction.
     * @param example The input example.
     */
    public Prediction(Prediction<T> other, int numUsed, Example<T> example) {
        this(other.output,new LinkedHashMap<>(other.outputScores),numUsed,example.size(),example,other.probability);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    @SuppressWarnings({"rawtypes","unchecked"}) // types are checked via getClass to ensure that the example, output and scores are all the same class.
    public static Prediction<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        PredictionImplProto proto = message.unpack(PredictionImplProto.class);
        int numUsed = proto.getNumUsed();
        if (numUsed < 0) {
            throw new IllegalStateException("Invalid protobuf, used a negative number of features");
        }
        int exampleSize = proto.getExampleSize();
        if (exampleSize < 0) {
            throw new IllegalStateException("Invalid protobuf, found a negative example size");
        }
        Example<?> example = ProtoUtil.deserialize(proto.getExample());
        Output<?> output = ProtoUtil.deserialize(proto.getOutput());
        if (!output.getClass().equals(example.getOutput().getClass())) {
            throw new IllegalStateException("Invalid protobuf, example and output types do not match, example = " + example.getOutput().getClass() + ", output = " + output.getClass());
        }
        Map map = new HashMap();
        for (Map.Entry<String, OutputProto> e : proto.getOutputScoresMap().entrySet()) {
            Output<?> tmpOutput = ProtoUtil.deserialize(e.getValue());
            if (!tmpOutput.getClass().equals(output.getClass())) {
                throw new IllegalStateException("Invalid protobuf, output scores not all the same type, found " + tmpOutput.getClass() + ", expected " + output.getClass());
            }
            map.put(e.getKey(), tmpOutput);
        }
        return new Prediction(output, map, numUsed, exampleSize, example, proto.getProbability());
    }

    /**
     * Returns the predicted output.
     * @return The predicted output.
    */
    public T getOutput() {
        return output;
    }

    /**
     * Returns the number of features used in the prediction.
     * @return The number of features used.
     */
    public int getNumActiveFeatures() {
        return numUsed;
    }

    /**
     * Returns the number of features in the example.
     * @return The number of features in the example.
     */
    public int getExampleSize() {
        return exampleSize;
    }

    /**
     * Returns the example itself.
     * @return The example.
     */
    public Example<T> getExample() {
        return example;
    }

    /**
     * Gets the output scores for each output.
     * <p>
     * May be an empty map if it did not generate scores.
     * @return A Map.
     */
    public Map<String,T> getOutputScores() {
        return outputScores;
    }

    /**
     * Are the scores probabilities?
     * @return True if the scores are probabilities.
     */
    public boolean hasProbabilities() {
        return probability;
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("Prediction(maxLabel=");
        buffer.append(output);
        buffer.append(",outputScores={");
        for (Map.Entry<String,T> e : outputScores.entrySet()) {
            buffer.append(e.toString());
            buffer.append(",");
        }
        buffer.delete(buffer.length()-1,buffer.length());
        buffer.append("})");

        return buffer.toString();
    }

    /**
     * Checks that the other prediction has the same distribution as this prediction,
     * using the {@link Output#fullEquals} method.
     * @param other The prediction to compare.
     * @return True if they have the same distributions.
     */
    public boolean distributionEquals(Prediction<T> other) {
        return distributionEquals(other, 0.0);
    }

    /**
     * Checks that the other prediction has the same distribution as this prediction,
     * using the {@link Output#fullEquals} method.
     * @param other The prediction to compare.
     * @param threshold The tolerance threshold for the scores.
     * @return True if they have the same distributions.
     */
    public boolean distributionEquals(Prediction<T> other, double threshold) {
        if (outputScores.size() != other.outputScores.size()) {
            return false;
        }
        for (Map.Entry<String,T> e : outputScores.entrySet()) {
            T otherScore = other.outputScores.get(e.getKey());
            if (otherScore == null) {
                return false;
            } else if (!e.getValue().fullEquals(otherScore, threshold)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public PredictionProto serialize() {
        return ProtoUtil.serialize(this);
    }
}
