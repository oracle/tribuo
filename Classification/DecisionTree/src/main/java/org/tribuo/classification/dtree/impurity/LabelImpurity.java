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

package org.tribuo.classification.dtree.impurity;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;

import java.util.Map;

/**
 * Calculates a tree impurity score based on label counts, weighted label counts or a probability distribution.
 * <p>
 * Impurity scores should be non-negative, with a zero signifying that the node is pure (i.e., contains
 * only a single label), and increasing values indicating an increasingly uniform distribution over possible labels.
 */
public interface LabelImpurity extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Calculates the impurity, assuming it's input is a normalized probability distribution.
     * @param input The input probability distribution.
     * @return The impurity.
     */
    public double impurityNormed(double[] input);

    /**
     * Calculates the impurity assuming the inputs are weighted counts normalizing by their sum.
     * @param input The input counts.
     * @return The impurity.
     */
    default public double impurityWeighted(double[] input) {
        double[] prob = new double[input.length];

        double sum = 0.0;
        for (double i : input) {
            sum += i;
        }

        for (int i = 0; i < input.length; i++) {
            prob[i] = input[i] / sum;
        }

        return sum*impurityNormed(prob);
    }

    /**
     * Calculates the impurity assuming the inputs are counts.
     * @param input The input counts.
     * @return The impurity.
     */
    default public double impurity(double[] input) {
        double[] prob = new double[input.length];

        double sum = 0.0;
        for (double i : input) {
            sum += i;
        }

        for (int i = 0; i < input.length; i++) {
            prob[i] = input[i] / sum;
        }

        return impurityNormed(prob);
    }

    /**
     * Calculates the impurity by assuming the input are weighted counts and converting them into a probability
     * distribution by dividing by their sum. The resulting impurity is then rescaled by multiplying by the sum.
     * @param input The input counts.
     * @return The impurity.
     */
    default public double impurityWeighted(float[] input) {
        double[] prob = new double[input.length];

        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            float f = input[i];
            sum += f;
        }

        for (int i = 0; i < input.length; i++) {
            prob[i] = input[i] / sum;
        }

        return sum*impurityNormed(prob);
    }

    /**
     * Calculates the impurity assuming the input are fractional counts.
     * @param input The input counts.
     * @return The impurity.
     */
    default public double impurity(float[] input) {
        double[] prob = new double[input.length];

        double sum = 0.0;
        for (double i : input) {
            sum += i;
        }

        for (int i = 0; i < input.length; i++) {
            prob[i] = input[i] / sum;
        }

        return impurityNormed(prob);
    }

    /**
     * Calculates the impurity assuming the input are counts.
     * @param input The input counts.
     * @return The impurity.
     */
    default public double impurity(int[] input) {
        double[] prob = new double[input.length];

        int sum = 0;
        for (int i : input) {
            sum += i;
        }

        double sumFloat = sum;
        for (int i = 0; i < input.length; i++) {
            prob[i] = input[i] / sumFloat;
        }

        return impurityNormed(prob);
    }

    /**
     * Takes a {@link Map} for weighted counts. Normalizes into a probability distribution before calling impurityNormed(double[]).
     * @param counts A map from instances to weighted counts.
     * @return The impurity score.
     */
    default public double impurity(Map<String,Double> counts) {
        double[] prob = new double[counts.size()];

        double sum = 0.0;
        int i = 0;
        for (Map.Entry<String,Double> e : counts.entrySet()) {
            sum += e.getValue();
            prob[i] = e.getValue();
            i++;
        }

        for (int j = 0; j < prob.length; j++) {
            prob[j] /= sum;
        }

        return impurityNormed(prob);
    }

}
