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

package org.tribuo.regression.sgd;

import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;

import java.util.SplittableRandom;

/**
 * Utilities. Currently stores methods for shuffling examples and their associated regression dimensions and weights.
 */
public class Util {
    /**
     * In place shuffle of the features, labels and weights.
     * @param features Input features.
     * @param regressors Input regressors.
     * @param weights Input weights.
     * @param rng SplittableRandom number generator.
     */
    public static void shuffleInPlace(SparseVector[] features, DenseVector[] regressors, double[] weights, SplittableRandom rng) {
        int size = features.length;
        // Shuffle array
        for (int i = size; i > 1; i--) {
            int j = rng.nextInt(i);
            //swap features
            SparseVector tmpFeature = features[i-1];
            features[i-1] = features[j];
            features[j] = tmpFeature;
            //swap regressors
            DenseVector tmpRegressors = regressors[i-1];
            regressors[i-1] = regressors[j];
            regressors[j] = tmpRegressors;
            //swap weights
            double tmpWeight = weights[i-1];
            weights[i-1] = weights[j];
            weights[j] = tmpWeight;
        }
    }

    /**
     * In place shuffle of the features, labels and weights.
     * @param features Input features.
     * @param regressors Input regressors.
     * @param weights Input weights.
     * @param indices Input indices.
     * @param rng SplittableRandom number generator.
     */
    public static void shuffleInPlace(SparseVector[] features, DenseVector[] regressors, double[] weights, int[] indices, SplittableRandom rng) {
        int size = features.length;
        // Shuffle array
        for (int i = size; i > 1; i--) {
            int j = rng.nextInt(i);
            //swap features
            SparseVector tmpFeature = features[i-1];
            features[i-1] = features[j];
            features[j] = tmpFeature;
            //swap regressors
            DenseVector tmpLabel = regressors[i-1];
            regressors[i-1] = regressors[j];
            regressors[j] = tmpLabel;
            //swap weights
            double tmpWeight = weights[i-1];
            weights[i-1] = weights[j];
            weights[j] = tmpWeight;
            //swap indices
            int tmpIndex = indices[i-1];
            indices[i-1] = indices[j];
            indices[j] = tmpIndex;
        }
    }
}
