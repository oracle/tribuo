/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.sgd;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.FeedForwardParameters;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.math.util.HeapMerger;
import org.tribuo.math.util.Merger;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;

/**
 * A {@link Parameters} for factorization machines.
 */
public final class FMParameters implements FeedForwardParameters {
    private static final long serialVersionUID = 1L;

    private static final Merger merger = new HeapMerger();

    private Tensor[] weights;
    private DenseVector biasVector;
    private DenseMatrix weightMatrix;

    private final int numFactors;

    /**
     * Constructor. The number of features and the number of outputs must be fixed and known in advance.
     *
     * @param rng         The RNG to use for initialization.
     * @param numFeatures The number of features in the training dataset.
     * @param numLabels   The number of outputs in the training dataset.
     * @param numFactors  The size of the factorized feature representation.
     * @param variance    The variance of the factor initializer.
     */
    public FMParameters(SplittableRandom rng, int numFeatures, int numLabels, int numFactors, double variance) {
        weights = new Tensor[numLabels + 2];
        biasVector = new DenseVector(numLabels);
        weightMatrix = new DenseMatrix(numLabels, numFeatures);
        weights[0] = biasVector;
        weights[1] = weightMatrix;
        for (int i = 0; i < numLabels; i++) {
            DenseMatrix curMatrix = new DenseMatrix(numFactors, numFeatures);
            initializeMatrix(rng, variance, curMatrix);
            weights[i + 2] = curMatrix;
        }
        this.numFactors = numFactors;
    }

    /**
     * Constructs a FMParameters wrapped around the weight array.
     *
     * @param weights    The weights to wrap.
     * @param numFactors The size of the factorized feature representation.
     */
    private FMParameters(Tensor[] weights, int numFactors) {
        this.weights = weights;
        this.biasVector = (DenseVector) weights[0];
        this.weightMatrix = (DenseMatrix) weights[1];
        this.numFactors = numFactors;
    }

    /**
     * Initializes the weight matrix by drawing from a zero mean gaussian with the specified variance.
     *
     * @param rng      The RNG to use.
     * @param variance The variance of the gaussian.
     * @param matrix   The matrix to initialize.
     */
    private void initializeMatrix(SplittableRandom rng, double variance, DenseMatrix matrix) {
        // This is to get a nextGaussian method. In Java 17 we can use the SplittableRandom directly
        // though that will prevent reproducibility as there will be different draws from the RNG and
        // the PRNG algorithm is different.
        Random innerRNG = new Random(rng.nextLong());
        int dim1 = matrix.getDimension1Size();
        int dim2 = matrix.getDimension2Size();
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                matrix.set(i, j, innerRNG.nextGaussian() * variance);
            }
        }
    }

    /**
     * Generates an unnormalised prediction by multiplying the weights with the incoming features,
     * adding the bias and adding the feature factors.
     *
     * @param example A feature vector
     * @return A {@link DenseVector} containing a score for each label.
     */
    @Override
    public DenseVector predict(SGDVector example) {
        // Linear part of the prediction
        DenseVector pred = weightMatrix.leftMultiply(example);

        // Add in the label biases
        pred.intersectAndAddInPlace(biasVector);

        // Compute the contribution of the feature factors
        DenseVector factorizedPred = new DenseVector(biasVector.size());
        for (int i = 2; i < weights.length; i++) {
            DenseMatrix curMatrix = (DenseMatrix) weights[i];
            double curValue = 0.0;
            for (int k = 0; k < numFactors; k++) {
                double sumOfSquares = 0.0;
                double sum = 0.0;
                for (VectorTuple v : example) {
                    double curWeight = curMatrix.get(k, v.index);
                    double value = curWeight * v.value;
                    sum += value;
                    sumOfSquares += value * value;
                }
                curValue += (sum * sum) - sumOfSquares;
            }
            curValue = curValue / 2;
            factorizedPred.set(i - 2, curValue);
        }

        // Add factorized portion
        pred.intersectAndAddInPlace(factorizedPred);
        return pred;
    }

    /**
     * Generate the gradients for a particular feature vector given
     * the loss and the per output gradients.
     * <p>
     * This method returns a {@link Tensor} array with numLabels + 2 elements.
     *
     * @param score    The Pair returned by the objective.
     * @param features The feature vector.
     * @return A {@link Tensor} array containing all the gradients.
     */
    @Override
    public Tensor[] gradients(Pair<Double, SGDVector> score, SGDVector features) {
        Tensor[] gradients = new Tensor[weights.length];

        SGDVector outputGradient = score.getB();
        // Bias gradient
        if (outputGradient instanceof SparseVector) {
            gradients[0] = ((SparseVector) outputGradient).densify();
        } else {
            gradients[0] = outputGradient.copy();
        }

        // Feature gradients
        gradients[1] = outputGradient.outer(features);

        // factorised representation gradients
        // per label
        for (int i = 2; i < weights.length; i++) {
            double curOutputGradient = outputGradient.get(i - 2);
            DenseMatrix curFactors = (DenseMatrix) weights[i];
            if (curOutputGradient != 0.0) {
                // compute /sum_j v_{j,f}x_j
                SGDVector factorSum = curFactors.leftMultiply(features);

                // grad_f: dy/d0 * (x_i * factorSum_f - v_{i,f} * x_i * x_i)
                Matrix factorGradMatrix;
                if (features instanceof SparseVector) {
                    List<SparseVector> vectors = new ArrayList<>(numFactors);
                    for (int j = 0; j < numFactors; j++) {
                        vectors.add(((SparseVector) features).copy());
                    }
                    factorGradMatrix = new DenseSparseMatrix(vectors);
                } else {
                    factorGradMatrix = new DenseMatrix(numFactors, features.size());
                    for (int j = 0; j < numFactors; j++) {
                        for (int k = 0; k < features.size(); k++) {
                            factorGradMatrix.set(j, k, features.get(k));
                        }
                    }
                }
                for (int j = 0; j < numFactors; j++) {
                    // This gets a mutable view of the row
                    SGDVector curFactorGrad = factorGradMatrix.getRow(j);
                    double curFactorSum = factorSum.get(j);
                    final int jFinal = j;

                    // Compute the gradient for this element of the factor vector
                    curFactorGrad.foreachIndexedInPlace((Integer idx, Double a) -> a * curFactorSum - curFactors.get(jFinal, idx) * a * a);

                    // Multiply by the output gradient
                    curFactorGrad.scaleInPlace(curOutputGradient);
                }
                gradients[i] = factorGradMatrix;
            } else {
                // If the output gradient is 0.0 then all the factor gradients are zero.
                // Technically with regularization we should shrink the weights for the specified features.
                gradients[i] = new DenseSparseMatrix(numFactors, features.size());
            }
        }

        return gradients;
    }

    /**
     * This returns a {@link DenseMatrix} the same size as the Parameters.
     *
     * @return A {@link Tensor} array containing a single {@link DenseMatrix}.
     */
    @Override
    public Tensor[] getEmptyCopy() {
        Tensor[] output = new Tensor[weights.length];
        output[0] = new DenseVector(biasVector.size());
        output[1] = new DenseMatrix(weightMatrix.getDimension1Size(), weightMatrix.getDimension2Size());
        for (int i = 2; i < weights.length; i++) {
            DenseMatrix curMatrix = (DenseMatrix) weights[i];
            output[i] = new DenseMatrix(curMatrix.getDimension1Size(), curMatrix.getDimension2Size());
        }
        return output;
    }

    @Override
    public Tensor[] get() {
        return weights;
    }

    @Override
    public void set(Tensor[] newWeights) {
        if (newWeights.length == weights.length) {
            weights = newWeights;
            biasVector = (DenseVector) weights[0];
            weightMatrix = (DenseMatrix) weights[1];
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
        Tensor[] output = new Tensor[weights.length];
        for (int i = 0; i < weights.length; i++) {
            if (gradients[0][i] instanceof DenseVector) {
                for (int j = 1; j < size; j++) {
                    gradients[0][i].intersectAndAddInPlace(gradients[j][i]);
                }
                output[i] = gradients[0][i];
            } else if (gradients[0][i] instanceof DenseMatrix) {
                for (int j = 1; j < size; j++) {
                    gradients[0][i].intersectAndAddInPlace(gradients[j][i]);
                }
                output[i] = gradients[0][i];
            } else if (gradients[0][i] instanceof DenseSparseMatrix) {
                DenseSparseMatrix[] updates = new DenseSparseMatrix[size];
                for (int j = 0; j < updates.length; j++) {
                    updates[j] = (DenseSparseMatrix) gradients[j][0];
                }

                DenseSparseMatrix update = merger.merge(updates);

                output[i] = update;
            } else {
                throw new IllegalStateException("Unexpected gradient type, expected DenseVector, DenseMatrix or DenseSparseMatrix, received " + gradients[0][i].getClass().getName());
            }
        }
        return output;
    }

    @Override
    public FMParameters copy() {
        Tensor[] weightCopy = new Tensor[weights.length];
        for (int i = 0; i < weights.length; i++) {
            weightCopy[i] = weights[i].copy();
        }
        return new FMParameters(weightCopy, numFactors);
    }
}
