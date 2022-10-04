/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.fs;

import org.tribuo.Dataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.classification.Label;

/**
 * Internal implementation interface for the feature selection module.
 * <p>
 * Not subject to the same API stability guarantees as the public classes of Tribuo.
 */
interface FSMatrix {

    /**
     * Builds an FSMatrix after binning the features into the specified number of bins.
     * @param dataset The dataset to use.
     * @param numBins The number of bins to use for the features.
     * @return An FSMatrix for computing information theoretic values.
     */
    static FSMatrix buildMatrix(Dataset<Label> dataset, int numBins) {
        return DenseFSMatrix.equalWidthBins(dataset, numBins);
    }

    /**
     * The number of features in this matrix.
     * @return The number of features.
     */
    int getNumFeatures();

    /**
     * The number of samples in this matrix.
     * @return The number of samples.
     */
    int getNumSamples();

    /**
     * The feature map for the wrapped dataset.
     * @return The feature map.
     */
    ImmutableFeatureMap getFeatureMap();

    /**
     * Calculates the mutual information I(X_i;Y) where Y is the label
     * @param featureIndex The index i
     * @return I(X_i;Y)
     */
    double mi(int featureIndex);

    /**
     * Calculates the mutual information I(X_i;X_j).
     * @param firstIndex The index i
     * @param secondIndex The index j
     * @return I(X_i;X_j)
     */
    double mi(int firstIndex, int secondIndex);

    /**
     * Generates an array of I(X_i;Y), ranging X_i over all possible features, where Y is the label.
     * @return An array where array[i] = I(X_i;Y)
     */
    default double[] miList() {
        int numFeatures = getNumFeatures();
        double[] row = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            row[i] = mi(i);
        }
        return row;
    }

    /**
     * Generates an array of I(X_i;X_j), ranging X_i over all possible features.
     * @param targetIndex The index of X_j variable
     * @return An array where array[i] = I(X_i;X_j)
     */
    default double[] miList(int targetIndex) {
        int numFeatures = getNumFeatures();
        double[] row = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            row[i] = mi(i,targetIndex);
        }
        return row;
    }

    /**
     * Calculates the joint mutual information I(X_i,X_j;Y) where Y is the label
     * @param featureIndex The index i
     * @param jointIndex The index j
     * @return I(X_iX_j;Y)
     */
    double jmi(int featureIndex, int jointIndex);

    /**
     * Calculates the joint mutual information I(X_i,X_j;X_k)
     * @param firstIndex The index i
     * @param secondIndex The index j
     * @param targetIndex The index k
     * @return I(X_i,X_j;X_k)
     */
    double jmi(int firstIndex, int secondIndex, int targetIndex);

    /**
     * Generates an array of I(X_i,Z;Y), ranging X_i over all possible features, where Y is the label.
     * @param jointIndex The index of the Z variable
     * @return An array where array[i] = I(X_i,Z;Y)
     */
    default double[] jmiList(int jointIndex) {
        int numFeatures = getNumFeatures();
        double[] row = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            row[i] = jmi(i,jointIndex);
        }
        return row;
    }

    /**
     * Generates an array of I(X_i,X_j;Y), ranging X_i over all possible features.
     * @param secondIndex The index of the X_j variable
     * @param targetIndex The index of the X_k variable
     * @return An array where array[i] = I(X_i,X_j;X_k)
     */
    default double[] jmiList(int secondIndex, int targetIndex) {
        int numFeatures = getNumFeatures();
        double[] row = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            row[i] = jmi(i,secondIndex,targetIndex);
        }
        return row;
    }

    /**
     * Calculates the conditional mutual information I(X_i;Y|X_j), where Y is the label
     * @param featureIndex The index i
     * @param conditionIndex The index j
     * @return I(X_i;Y|X_j)
     */
    double cmi(int featureIndex, int conditionIndex);

    /**
     * Calculates the conditional mutual information I(X_i;X_j|X_k)
     * @param firstIndex The index i
     * @param secondIndex The index j
     * @param conditionIndex The index k
     * @return I(X_i;X_j|X_k)
     */
    double cmi(int firstIndex, int secondIndex, int conditionIndex);

    /**
     * Generates an array of I(X_i;Y|Z), ranging X_i over all possible features, where Y is the label.
     * @param conditionIndex The index of the Z variable
     * @return An array where array[i] = I(X_i;Y|Z)
     */
    default double[] cmiList(int conditionIndex) {
        int numFeatures = getNumFeatures();
        double[] row = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            row[i] = cmi(i,conditionIndex);
        }
        return row;
    }

    /**
     * Generates an array of I(X_i;X_j|X_k), ranging X_i over all possible features.
     * @param featureIndex The index of the X_j variable
     * @param conditionIndex The index of the X_k variable
     * @return An array where array[i] = I(X_i;X_j|X_k)
     */
    default double[] cmiList(int featureIndex, int conditionIndex) {
        int numFeatures = getNumFeatures();
        double[] row = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            row[i] = cmi(i,featureIndex,conditionIndex);
        }
        return row;
    }
}
