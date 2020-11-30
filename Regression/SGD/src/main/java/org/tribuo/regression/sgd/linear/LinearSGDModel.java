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

package org.tribuo.regression.sgd.linear;

import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;

import java.util.Arrays;

/**
 * The inference time version of a linear model trained using SGD.
 * The output dimensions are independent, unless they are tied together by the
 * optimiser.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public class LinearSGDModel extends AbstractLinearSGDModel<Regressor> {
    private static final long serialVersionUID = 3L;

    private final String[] dimensionNames;

    LinearSGDModel(String name, String[] dimensionNames, ModelProvenance description,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> labelIDMap,
                          LinearParameters parameters) {
        super(name, description, featureIDMap, labelIDMap, parameters.getWeightMatrix(), false);
        this.dimensionNames = dimensionNames;
    }

    private LinearSGDModel(String name, String[] dimensionNames, ModelProvenance description,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> labelIDMap,
                          DenseMatrix weights) {
        super(name, description, featureIDMap, labelIDMap, weights, false);
        this.dimensionNames = dimensionNames;
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        PredAndActive predTuple = predictSingle(example);
        return new Prediction<>(new Regressor(dimensionNames,predTuple.prediction.toArray()), predTuple.numActiveFeatures-1, example);
    }

    @Override
    protected LinearSGDModel copy(String newName, ModelProvenance newProvenance) {
        return new LinearSGDModel(newName,Arrays.copyOf(dimensionNames,dimensionNames.length),newProvenance,featureIDMap,outputIDInfo,getWeightsCopy());
    }

    @Override
    protected String getDimensionName(int index) {
        return dimensionNames[index];
    }
}
