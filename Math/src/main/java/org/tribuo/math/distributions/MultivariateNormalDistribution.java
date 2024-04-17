/*
 * Copyright (c) 2022, 2024, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.Tensor;

import java.util.Arrays;
import java.util.Optional;
import java.util.Random;
import java.util.random.RandomGenerator;

/**
 * A class for sampling from multivariate normal distributions.
 */
public final class MultivariateNormalDistribution {

    private final long seed;
    private final Random rng;
    private final DenseVector means;
    private final double variance;
    private final DenseVector covarianceVector;
    private final DenseMatrix covarianceMatrix;
    private final DenseMatrix samplingCovariance;
    private final boolean eigenDecomposition;
    private final CovarianceType type;

    /**
     * Constructs a multivariate normal distribution that can be sampled from.
     * <p>
     * Throws {@link IllegalArgumentException} if the covariance matrix is not positive definite.
     * <p>
     * Uses a {@link DenseMatrix.CholeskyFactorization} to compute the sampling
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
     * Uses a {@link DenseMatrix.CholeskyFactorization} to compute the sampling
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
        this(means, covariance, CovarianceType.FULL, seed, eigenDecomposition);
    }

    /**
     * Constructs a multivariate normal distribution that can be sampled from.
     * <p>
     * Throws {@link IllegalArgumentException} if the covariance matrix is not positive definite.
     * @param means The mean vector.
     * @param covariance The covariance matrix. If type is {@link CovarianceType#FULL} must be a {@link DenseMatrix},
     *                   if {@link CovarianceType#DIAGONAL} or {@link CovarianceType#SPHERICAL} must be a
     *                   {@link DenseVector}. Spherical covariances should have a single element dense vector.
     * @param type The covariance type.
     * @param seed The RNG seed.
     * @param eigenDecomposition If true use an eigen decomposition to compute the sampling covariance matrix
     *                           rather than a cholesky factorization, if it's a full covariance.
     */
    public MultivariateNormalDistribution(DenseVector means, Tensor covariance, CovarianceType type, long seed, boolean eigenDecomposition) {
        this.seed = seed;
        this.rng = new Random(seed);
        this.means = means.copy();
        this.eigenDecomposition = eigenDecomposition;
        this.type = type;
        switch (type) {
            case FULL -> {
                if (!(covariance instanceof DenseMatrix)) {
                    throw new IllegalArgumentException("Covariance matrix must be a square matrix for full covariance, found " + covariance.getClass());
                }
                this.covarianceMatrix = (DenseMatrix) covariance.copy();
                if (this.covarianceMatrix.getDimension1Size() != this.means.size() || this.covarianceMatrix.getDimension2Size() != this.means.size()) {
                    throw new IllegalArgumentException("Covariance matrix must be square and the same dimension as the mean vector. Mean vector size = " + means.size() + ", covariance size = " + Arrays.toString(this.covarianceMatrix.getShape()));
                }
                if (eigenDecomposition) {
                    Optional<DenseMatrix.EigenDecomposition> factorization = this.covarianceMatrix.eigenDecomposition();
                    if (factorization.isPresent() && factorization.get().positiveEigenvalues()) {
                        DenseVector eigenvalues = factorization.get().eigenvalues();
                        // rows are eigenvectors
                        DenseMatrix eigenvectors = new DenseMatrix(factorization.get().eigenvectors());
                        // scale eigenvectors by sqrt of eigenvalues
                        eigenvalues.foreachInPlace(Math::sqrt);
                        DenseSparseMatrix diagonal = DenseSparseMatrix.createDiagonal(eigenvalues);
                        this.samplingCovariance = eigenvectors.matrixMultiply(diagonal).matrixMultiply(eigenvectors,false,true);
                    } else {
                        throw new IllegalArgumentException("Covariance matrix is not positive definite.");
                    }
                } else {
                    Optional<DenseMatrix.CholeskyFactorization> factorization = this.covarianceMatrix.choleskyFactorization();
                    if (factorization.isPresent()) {
                        this.samplingCovariance = factorization.get().lMatrix();
                    } else {
                        throw new IllegalArgumentException("Covariance matrix is not positive definite.");
                    }
                }
                // set unused variables.
                this.covarianceVector = null;
                this.variance = Double.NaN;
            }
            case DIAGONAL -> {
                if (!(covariance instanceof DenseVector)) {
                    throw new IllegalArgumentException("Covariance must be a vector for diagonal covariance, found " + covariance.getClass());
                }
                this.covarianceVector = (DenseVector) covariance.copy();
                if (this.covarianceVector.size() != this.means.size()) {
                    throw new IllegalArgumentException("Covariance must be a vector and the same dimension as the mean vector. Mean vector size = " + means.size() + ", covariance size = " + this.covarianceVector.size());
                }

                // set unused variables.
                this.covarianceMatrix = null;
                this.samplingCovariance = null;
                this.variance = Double.NaN;
            }
            case SPHERICAL -> {
                if (covariance instanceof DenseVector vec) {
                    if (vec.size() != 1) {
                        throw new IllegalArgumentException("Covariance must be a single element vector for spherical covariance. Found " + vec.size());
                    }
                } else {
                    throw new IllegalArgumentException("Covariance must be a single element vector for spherical covariance, found " + covariance.getClass());
                }
                this.variance = Double.NaN;

                // set unused variables.
                this.covarianceVector = null;
                this.covarianceMatrix = null;
                this.samplingCovariance = null;
            }
            default -> throw new IllegalArgumentException("Unknown covariance type " + type);
        }
    }

    /**
     * Sample a vector from this multivariate normal distribution.
     * @return A sample from this distribution.
     */
    public DenseVector sampleVector() {
        return sampleVector(rng);
    }

    /**
     * Sample a vector from this multivariate normal distribution.
     * @return A sample from this distribution.
     */
    public DenseVector sampleVector(RandomGenerator otherRNG) {
        DenseVector sampled = new DenseVector(means.size());
        for (int i = 0; i < means.size(); i++) {
            sampled.set(i, otherRNG.nextGaussian());
        }

        switch (type) {
            case FULL -> sampled = samplingCovariance.leftMultiply(sampled);
            case DIAGONAL -> sampled.hadamardProductInPlace(covarianceVector);
            case SPHERICAL -> sampled.scaleInPlace(variance);
        }

        return means.add(sampled);
    }

    /**
     * Sample a vector from this multivariate normal distribution.
     * @return A sample from this distribution.
     */
    public double[] sampleArray() {
        return sampleVector().toArray();
    }

    /**
     * Compute the log probability of the input under this multivariate normal distribution.
     * @param input The input to compute.
     * @return The log probability.
     */
    public double logProbability(SGDVector input) {

    }

    @Override
    public String toString() {
        return switch (type) {
            case FULL -> "MultivariateNormal(mean="+means+",covariance="+covarianceMatrix+",seed="+seed+",useEigenDecomposition="+eigenDecomposition+",type="+type+")";
            case DIAGONAL -> "MultivariateNormal(mean="+means+",covariance="+covarianceVector+",seed="+seed+",useEigenDecomposition="+eigenDecomposition+",type="+type+")";
            case SPHERICAL -> "MultivariateNormal(mean="+means+",covariance="+variance+",seed="+seed+",useEigenDecomposition="+eigenDecomposition+",type="+type+")";
        };
    }

    /**
     * Type of the covariance in a multivariate normal distribution.
     */
    public enum CovarianceType {
        /**
         * Full covariance.
         */
        FULL(0),
        /**
         * Diagonal covariance.
         */
        DIAGONAL(1),
        /**
         * Spherical covariance.
         */
        SPHERICAL(2);

        private final int value;
        private CovarianceType(int value) {
            this.value = value;
        }

        /**
         * The enum value used for serialization.
         * @return The enum value.
         */
        public int value() {
            return value;
        }

        public static CovarianceType fromValue(int value) {
            CovarianceType[] values = CovarianceType.values();
            for (CovarianceType t : values) {
                if (t.value == value) {
                    return t;
                }
            }
            // Failed to find the enum.
            throw new IllegalStateException("Invalid CovarianceType enum value, found " + value);
        }
    }
}
