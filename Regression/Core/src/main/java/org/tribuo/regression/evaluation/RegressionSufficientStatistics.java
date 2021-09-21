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

package org.tribuo.regression.evaluation;

import com.oracle.labs.mlrg.olcut.util.MutableDouble;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The sufficient statistics for regression metrics (i.e., each prediction and each true value).
 */
public final class RegressionSufficientStatistics {

    final int n;
    final ImmutableOutputInfo<Regressor> domain;

    final Map<String, MutableDouble> sumAbsoluteError = new LinkedHashMap<>();
    final Map<String, MutableDouble> sumSquaredError = new LinkedHashMap<>();

    final Map<String, double[]> predictedValues = new LinkedHashMap<>();
    final Map<String, double[]> trueValues = new LinkedHashMap<>();

    // if useExampleWeights is false, all weights are set to 1.0
    final float[] weights;

    // if useExampleWeights is false, weightSum == n
    final float weightSum;

    /**
     * Constructs the sufficient statistics for regression metrics.
     * @param domain The output domain.
     * @param predictions The predictions.
     * @param useExampleWeights Should example weights be used.
     */
    public RegressionSufficientStatistics(ImmutableOutputInfo<Regressor> domain, List<Prediction<Regressor>> predictions, boolean useExampleWeights) {
        this.domain = domain;
        this.n = predictions.size();
        this.weights = initWeights(predictions, useExampleWeights);
        for (Regressor e : domain.getDomain()) {
            String name = e.getNames()[0];
            sumAbsoluteError.put(name,new MutableDouble());
            sumSquaredError.put(name,new MutableDouble());
            predictedValues.put(name,new double[this.n]);
            trueValues.put(name,new double[this.n]);
        }
        this.weightSum = tabulate(predictions);
    }

    private float tabulate(List<Prediction<Regressor>> predictions) {
        float weightSum = 0f;

        for (int i = 0; i < this.n; i++) {
            Prediction<Regressor> prediction = predictions.get(i);

            float weight = weights[i];
            weightSum += weight;

            Regressor pred = prediction.getOutput();
            Regressor truth = prediction.getExample().getOutput();
            if (truth.equals(RegressionFactory.UNKNOWN_REGRESSOR)) {
                throw new IllegalArgumentException("The sentinel Unknown Regressor was used as a ground truth output at prediction number " + i);
            } else if (pred.equals(RegressionFactory.UNKNOWN_REGRESSOR)) {
                throw new IllegalArgumentException("The sentinel Unknown Regressor was predicted by the model at prediction number " + i);
            }

            for (int j = 0; j < truth.size(); j++) {
                String name = truth.getNames()[j];
                double trueValue = truth.getValues()[j];
                double predValue = pred.getValues()[j];

                double diff = trueValue - predValue;
                sumAbsoluteError.get(name).increment(weight*Math.abs(diff));
                sumSquaredError.get(name).increment(weight*diff*diff);

                trueValues.get(name)[i] = trueValue;
                predictedValues.get(name)[i] = predValue;
            }
        }
        return weightSum;
    }

    private static float[] initWeights(List<Prediction<Regressor>> predictions, boolean useExampleWeights) {
        float[] weights = new float[predictions.size()];
        if (useExampleWeights) {
            for (int i = 0; i < predictions.size(); i++) {
                weights[i] = predictions.get(i).getExample().getWeight();
            }
        } else {
            Arrays.fill(weights,1.0f);
        }
        return weights;
    }

}