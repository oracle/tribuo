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

package org.tribuo.math.distributions;

import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.DenseVector;

import java.util.Arrays;
import java.util.Optional;
import java.util.Random;

/**
 * A class for sampling from multivariate normal distributions.
 */
public final class MultivariateNormalDistribution {

    private final long seed;
    private final Random rng;
    private final DenseVector means;
    private final DenseMatrix covariance;
    private final DenseMatrix samplingCovariance;
    private final boolean eigenDecomposition;

    /**
     * Constructs a multivariate normal distribution that can be sampled from.
     * <p>
     * Throws {@link IllegalArgumentException} if the covariance matrix is not positive definite.
     * <p>
     * Uses a {@link org.tribuo.math.la.DenseMatrix.CholeskyFactorization} to compute the sampling
     * covariance matrix.
     * @param means The mean vector.
     * @param covariance The covariance matrix.
     * @param seed The RNG seed.
     */
    public MultivariateNormalDistribution(double[] means, double[][] covariance, long seed) {
        this(DenseVector.createDenseVector(means),DenseMatrix.createDenseMatrix(covariance),seed);
    }

    /**
     * Constructs a multivariate normal distribution that can be sampled from.
     * <p>
     * Throws {@link IllegalArgumentException} if the covariance matrix is not positive definite.
     * @param means The mean vector.
     * @param covariance The covariance matrix.
     * @param seed The RNG seed.
     * @param eigenDecomposition If true use an eigen decomposition to compute the sampling covariance matrix
     *                           rather than a cholesky factorization.
     */
    public MultivariateNormalDistribution(double[] means, double[][] covariance, long seed, boolean eigenDecomposition) {
        this(DenseVector.createDenseVector(means),DenseMatrix.createDenseMatrix(covariance),seed, eigenDecomposition);
    }

    /**
     * Constructs a multivariate normal distribution that can be sampled from.
     * <p>
     * Throws {@link IllegalArgumentException} if the covariance matrix is not positive definite.
     * <p>
     * Uses a {@link org.tribuo.math.la.DenseMatrix.CholeskyFactorization} to compute the sampling
     * covariance matrix.
     * @param means The mean vector.
     * @param covariance The covariance matrix.
     * @param seed The RNG seed.
     */
    public MultivariateNormalDistribution(DenseVector means, DenseMatrix covariance, long seed) {
        this(means,covariance,seed,false);
    }

    /**
     * Constructs a multivariate normal distribution that can be sampled from.
     * <p>
     * Throws {@link IllegalArgumentException} if the covariance matrix is not positive definite.
     * @param means The mean vector.
     * @param covariance The covariance matrix.
     * @param seed The RNG seed.
     * @param eigenDecomposition If true use an eigen decomposition to compute the sampling covariance matrix
     *                           rather than a cholesky factorization.
     */
    public MultivariateNormalDistribution(DenseVector means, DenseMatrix covariance, long seed, boolean eigenDecomposition) {
        this.seed = seed;
        this.rng = new Random(seed);
        this.means = means.copy();
        this.covariance = covariance.copy();
        if (this.covariance.getDimension1Size() != this.means.size() || this.covariance.getDimension2Size() != this.means.size()) {
            throw new IllegalArgumentException("Covariance matrix must be square and the same dimension as the mean vector. Mean vector size = " + means.size() + ", covariance size = " + Arrays.toString(this.covariance.getShape()));
        }
        this.eigenDecomposition = eigenDecomposition;
        if (eigenDecomposition) {
            Optional<DenseMatrix.EigenDecomposition> factorization = this.covariance.eigenDecomposition();
            if (factorization.isPresent() && factorization.get().positiveEigenvalues()) {
                DenseVector eigenvalues = factorization.get().eigenvalues();
                // rows are eigenvectors
                DenseMatrix eigenvectors = new DenseMatrix(factorization.get().eigenvectors());
                // scale eigenvectors by sqrt of eigenvalues
                eigenvalues.foreachInPlace(Math::sqrt);
                DenseSparseMatrix diagonal = DenseSparseMatrix.createDiagonal(eigenvalues);;
                this.samplingCovariance = eigenvectors.matrixMultiply(diagonal).matrixMultiply(eigenvectors,false,true);
            } else {
                throw new IllegalArgumentException("Covariance matrix is not positive definite.");
            }
        } else {
            Optional<DenseMatrix.CholeskyFactorization> factorization = this.covariance.choleskyFactorization();
            if (factorization.isPresent()) {
                this.samplingCovariance = factorization.get().lMatrix();
            } else {
                throw new IllegalArgumentException("Covariance matrix is not positive definite.");
            }
        }
    }

    /**
     * Sample a vector from this multivariate normal distribution.
     * @return A sample from this distribution.
     */
    public DenseVector sampleVector() {
        DenseVector sampled = new DenseVector(means.size());
        for (int i = 0; i < means.size(); i++) {
            sampled.set(i,rng.nextGaussian());
        }

        sampled = samplingCovariance.leftMultiply(sampled);

        return means.add(sampled);
    }

    /**
     * Sample a vector from this multivariate normal distribution.
     * @return A sample from this distribution.
     */
    public double[] sampleArray() {
        return sampleVector().toArray();
    }

    @Override
    public String toString() {
        return "MultivariateNormal(mean="+means+",covariance="+covariance+",seed="+seed+",useEigenDecomposition="+eigenDecomposition+")";
    }
}
