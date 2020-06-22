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

package org.tribuo.math;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.util.HeapMerger;
import org.tribuo.math.util.Merger;

/**
 * A Parameters for producing single label linear models.
 */
public class LinearParameters implements Parameters {

    private static final Merger merger = new HeapMerger();

    // Last row in this DenseMatrix is the bias, added by calling new SparseVector(example,featureInfo,true);
    private Tensor[] weights;
    private DenseMatrix weightMatrix;

    /**
     * Constructor. The number of features and the number of outputs must be fixed and known in advance.
     * @param numFeatures The number of features in the training dataset (excluding the bias).
     * @param numLabels The number of outputs in the training dataset.
     */
    public LinearParameters(int numFeatures, int numLabels) {
        weights = new Tensor[1];
        weightMatrix = new DenseMatrix(numLabels,numFeatures);
        weights[0] = weightMatrix;
    }

    /**
     * Generates an unnormalised prediction by leftMultiply'ing the weights with the incoming features.
     * @param example A feature vector
     * @return A {@link org.tribuo.math.la.DenseVector} containing a score for each label.
     */
    public SGDVector predict(SparseVector example) {
        return weightMatrix.leftMultiply(example);
    }

    /**
     * Generate the gradients for a particular feature vector given
     * the loss and the per output gradients.
     *
     * This parameters returns a single element {@link Tensor} array.
     * @param score The Pair returned by the objective.
     * @param features The feature vector.
     * @return A {@link Tensor} array with a single {@link DenseSparseMatrix} containing all gradients.
     */
    public Tensor[] gradients(Pair<Double, SGDVector> score, SparseVector features) {
        Tensor[] output = new Tensor[1];
        output[0] = score.getB().outer(features);
        return output;
    }

    /**
     * This returns a {@link DenseMatrix} the same size as the Parameters.
     * @return A {@link Tensor} array containing a single {@link DenseMatrix}.
     */
    @Override
    public Tensor[] getEmptyCopy() {
        DenseMatrix matrix = new DenseMatrix(weightMatrix.getDimension1Size(),weightMatrix.getDimension2Size());
        Tensor[] output = new Tensor[1];
        output[0] = matrix;
        return output;
    }

    @Override
    public Tensor[] get() {
        return weights;
    }

    /**
     * Returns the weight matrix.
     * @return The weight matrix.
     */
    public DenseMatrix getWeightMatrix() {
        return weightMatrix;
    }

    @Override
    public void set(Tensor[] newWeights) {
        if (newWeights.length == weights.length) {
            weights = newWeights;
            weightMatrix = (DenseMatrix) weights[0];
        }
    }

    @Override
    public void update(Tensor[] gradients) {
        for (int i = 0; i < gradients.length; i++) {
            weights[i].intersectAndAddInPlace(gradients[i]);
        }
    }

    @Override
    public Tensor[] merge(Tensor[][] gradients, int size) {
        DenseSparseMatrix[] updates = new DenseSparseMatrix[size];
        for (int j = 0; j < updates.length; j++) {
            updates[j] = (DenseSparseMatrix) gradients[j][0];
        }

        DenseSparseMatrix update = merger.merge(updates);

        return new Tensor[]{update};
    }
}
