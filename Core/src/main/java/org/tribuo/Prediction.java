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

package org.tribuo;

import java.io.Serializable;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A prediction made by a {@link Model}.
 * Contains the output, and optionally and a map of scores over the possible outputs.
 * If hasProbabilities() == true then it has a probability
 * distribution over outputs otherwise it is unnormalized scores over outputs.
 * <p>
 * If possible it also contains the number of features that were used to make a prediction,
 * and how many features originally existed in the {@link Example}.
 */
public class Prediction<T extends Output<T>> implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * The example which was used to generate this prediction.
     */
    private final Example<T> example;

    /**
     * The output assigned by a classifier.
     */
    private final T output;

    /**
     * Does outputScores contain probabilities or scores?
     */
    private final boolean probability;

    /**
     * How many features were used by the model.
     */
    private final int numUsed;

    /**
     * How many features were set in the example.
     */
    private final int exampleSize;

    /**
     * A map from output name to output object, which contains the score.
     */
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
        }
        buffer.delete(buffer.length()-2,buffer.length());
        buffer.append("})");

        return buffer.toString();
    }
}
