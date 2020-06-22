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

package org.tribuo.regression.ensemble;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.regression.Regressor;

import java.util.Arrays;
import java.util.List;

/**
 * A combiner which performs a weighted or unweighted average of the predicted
 * regressors independently across the output dimensions.
 */
public class AveragingCombiner implements EnsembleCombiner<Regressor> {
    private static final long serialVersionUID = 1L;

    @Override
    public Prediction<Regressor> combine(ImmutableOutputInfo<Regressor> outputInfo, List<Prediction<Regressor>> predictions) {
        int numPredictions = predictions.size();
        int dimensions = outputInfo.size();
        int numUsed = 0;
        String[] names;
        double[] mean = new double[dimensions];
        double[] variance = new double[dimensions];
        for (Prediction<Regressor> p : predictions) {
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            Regressor curValue = p.getOutput();
            for (int i = 0; i < dimensions; i++) {
                double value = curValue.getValues()[i];
                double oldMean = mean[i];
                mean[i] += (value - oldMean);
                variance[i] += (value - oldMean) * (value - mean[i]);
            }
        }
        names = predictions.get(0).getOutput().getNames();
        if (numPredictions > 1) {
            for (int i = 0; i < dimensions; i++) {
                variance[i] /= (numPredictions-1);
            }
        } else {
            Arrays.fill(variance,0);
        }

        Example<Regressor> example = predictions.get(0).getExample();
        return new Prediction<>(new Regressor(names,mean,variance),numUsed,example);
    }

    @Override
    public Prediction<Regressor> combine(ImmutableOutputInfo<Regressor> outputInfo, List<Prediction<Regressor>> predictions, float[] weights) {
        if (predictions.size() != weights.length) {
            throw new IllegalArgumentException("predictions and weights must be the same length. predictions.size()="+predictions.size()+", weights.length="+weights.length);
        }
        int dimensions = outputInfo.size();
        int numUsed = 0;
        String[] names;
        double[] mean = new double[dimensions];
        double[] variance = new double[dimensions];
        double weightSum = 0.0;
        for (int i = 0; i < weights.length; i++) {
            Prediction<Regressor> p = predictions.get(i);
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            Regressor curValue = p.getOutput();
            float weight = weights[i];
            weightSum += weight;
            for (int j = 0; j < dimensions; j++) {
                double value = curValue.getValues()[j];
                double oldMean = mean[j];
                mean[j] += (weight / weightSum) * (value - oldMean);
                variance[j] += weight * (value - oldMean) * (value - mean[j]);
            }
        }
        names = predictions.get(0).getOutput().getNames();
        if (weights.length > 1) {
            for (int i = 0; i < dimensions; i++) {
                variance[i] /= (weightSum-1);
            }
        } else {
            Arrays.fill(variance,0);
        }

        Example<Regressor> example = predictions.get(0).getExample();
        return new Prediction<>(new Regressor(names,mean,variance),numUsed,example);
    }

    @Override
    public String toString() {
        return "MultipleOutputAveragingCombiner()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"EnsembleCombiner");
    }
}
