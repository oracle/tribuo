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

package org.tribuo.regression.xgboost;

import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.common.xgboost.XGBoostOutputConverter;
import org.tribuo.regression.Regressor;

import java.util.ArrayList;
import java.util.List;

/**
 * Converts XGBoost outputs into {@link Regressor} {@link Prediction}s.
 * <p>
 * Instances of this class are stateless and thread-safe.
 */
public final class XGBoostRegressionConverter implements XGBoostOutputConverter<Regressor> {
    private static final long serialVersionUID = 1L;

    /**
     * Construct an XGBoostRegressionConverter.
     */
    public XGBoostRegressionConverter() {}

    @Override
    public boolean generatesProbabilities() {
        return false;
    }

    @Override
    public Prediction<Regressor> convertOutput(ImmutableOutputInfo<Regressor> info, List<float[]> probabilities, int numValidFeatures, Example<Regressor> example) {
        Regressor.DimensionTuple[] tuples = new Regressor.DimensionTuple[probabilities.size()];
        int i = 0;
        for (float[] f : probabilities) {
            tuples[i] = new Regressor.DimensionTuple(info.getOutput(i).getNames()[0],f[0]);
            i++;
        }
        return new Prediction<>(new Regressor(tuples),numValidFeatures,example);
    }

    @Override
    public List<Prediction<Regressor>> convertBatchOutput(ImmutableOutputInfo<Regressor> info, List<float[][]> probabilities, int[] numValidFeatures, Example<Regressor>[] examples) {
        if ((numValidFeatures.length != examples.length) || (probabilities.get(0).length != numValidFeatures.length)) {
            throw new IllegalArgumentException("Lengths not the same, numValidFeatures.length = "
                    + numValidFeatures.length + ", examples.length = " + examples.length
                    + ", probabilities.get(0).length = " + probabilities.get(0).length);
        }
        Regressor.DimensionTuple[][] tuples = new Regressor.DimensionTuple[numValidFeatures.length][probabilities.size()];
        for (int i = 0; i < probabilities.size(); i++) {
            float[][] f = probabilities.get(i);
            String curName = info.getOutput(i).getNames()[0];
            for (int j = 0; j < numValidFeatures.length; j++) {
                tuples[j][i] = new Regressor.DimensionTuple(curName, f[j][0]);
            }
        }
        List<Prediction<Regressor>> predictions = new ArrayList<>();
        for (int i = 0; i < numValidFeatures.length; i++) {
            predictions.add(new Prediction<>(new Regressor(tuples[i]),numValidFeatures[i],examples[i]));
        }
        return predictions;
    }

}
