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

import com.oracle.labs.mlrg.olcut.util.Pair;

import java.util.List;
import java.util.Map;

/**
 * Holds an {@link Example}, a {@link Prediction} and a Map from String to List of Pairs
 * that contains the per output explanation.
 */
public class Excuse<T extends Output<T>> {

    private final Example<T> example;
    private final Prediction<T> prediction;
    private final Map<String,List<Pair<String,Double>>> weights;

    /**
     * Constructs an excuse for the prediction of the supplied example, using the feature weights.
     * @param example The example to excuse.
     * @param prediction The prediction to excuse.
     * @param weights The feature weights involved.
     */
    public Excuse(Example<T> example, Prediction<T> prediction, Map<String,List<Pair<String,Double>>> weights) {
        this.example = example;
        this.prediction = prediction;
        this.weights = weights;
    }

    /**
     * Returns the features involved in this excuse.
     * @param label The features for the requested output.
     * @return The features invovled.
     */
    public List<Pair<String,Double>> excuse(String label) {
        return weights.get(label);
    }

    /**
     * Returns the prediction being excused.
     * @return The prediction.
     */
    public Prediction<T> getPrediction() {
        return prediction;
    }

    /**
     * Returns the scores for all outputs and the relevant feature values.
     * @return The output scores and feature values.
     */
    public Map<String,List<Pair<String,Double>>> getScores() {
        return weights;
    }

    /**
     * The example being excused.
     * @return The example.
     */
    public Example<T> getExample() {
        return example;
    }

}
