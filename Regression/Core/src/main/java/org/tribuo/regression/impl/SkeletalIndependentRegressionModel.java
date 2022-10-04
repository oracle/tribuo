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

package org.tribuo.regression.impl;

import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;

import java.util.Arrays;

/**
 * A {@link Model} which wraps n independent regression models, where n is the
 * size of the MultipleRegressor domain. Each model independently predicts
 * a single regression dimension.
 */
public abstract class SkeletalIndependentRegressionModel extends Model<Regressor> {
    private static final long serialVersionUID = 1L;

    /**
     * The output dimension names.
     */
    protected final String[] dimensions;

    /**
     * models.size() must equal labelInfo.getDomain().size()
     * @param name Model name.
     * @param dimensions Dimension names.
     * @param modelProvenance The model provenance.
     * @param featureMap The feature domain used in training.
     * @param outputInfo The output domain used in training.
     */
    protected SkeletalIndependentRegressionModel(String name, String[] dimensions, ModelProvenance modelProvenance, ImmutableFeatureMap featureMap, ImmutableOutputInfo<Regressor> outputInfo) {
        super(name,modelProvenance,featureMap,outputInfo,false);
        this.dimensions = Arrays.copyOf(dimensions,dimensions.length);
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        SparseVector features = createFeatures(example);

        Regressor.DimensionTuple[] outputs = new Regressor.DimensionTuple[dimensions.length];

        for (int i = 0; i < dimensions.length; i++) {
            outputs[i] = scoreDimension(i,features);
        }

        return new Prediction<>(new Regressor(outputs),features.numActiveElements(),example);
    }

    /**
     * Creates the feature vector. Does not include a bias term.
     * <p>
     * Designed to be overridden.
     * @param example The example to convert.
     * @return The feature vector.
     */
    protected SparseVector createFeatures(Example<Regressor> example) {
        return SparseVector.createSparseVector(example,featureIDMap,false);
    }

    /**
     * Makes a prediction for a single dimension.
     * @param dimensionIdx The dimension index to predict.
     * @param features The features to use.
     * @return A single dimension prediction.
     */
    protected abstract Regressor.DimensionTuple scoreDimension(int dimensionIdx, SparseVector features);
}
