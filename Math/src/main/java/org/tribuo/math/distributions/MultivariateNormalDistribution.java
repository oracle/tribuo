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
public final class MultivariateNormalDistribution implements Distribution {

    private final long seed;
    private final Random rng;
    private final DenseVector means;
    private final double variance;
    private final DenseVector covarianceVector;
    private final DenseMatrix covarianceMatrix;
    private final DenseMatrix samplingCovariance;
    private final Tensor precision;
    private final double determinant;
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
     * Constructs a multivariate normal distribution that can be sampled from using a spherical covariance.
     * @param means The mean vector.
     * @param sphericalCovariance The spherical covariance matrix, stored as a single double.
     * @param seed The RNG seed.
     */
    public MultivariateNormalDistribution(DenseVector means, double sphericalCovariance, long seed) {
        this(means,new DenseVector(1,sphericalCovariance),CovarianceType.SPHERICAL,seed,false);
    }

    /**
     * Constructs a multivariate normal distribution that can be sampled from using a diagonal covariance matrix.
     * @param means The mean vector.
     * @param diagonalCovariance The diagonal covariance matrix, stored as a vector.
     * @param seed The RNG seed.
     */
    public MultivariateNormalDistribution(DenseVector means, DenseVector diagonalCovariance, long seed) {
        this(means,diagonalCovariance,CovarianceType.DIAGONAL,seed,false);
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
                        this.determinant = factorization.get().determinant();
                        this.precision = factorization.get().inverse();
                    } else {
                        throw new IllegalArgumentException("Covariance matrix is not positive definite.");
                    }
                } else {
                    Optional<DenseMatrix.CholeskyFactorization> factorization = this.covarianceMatrix.choleskyFactorization();
                    if (factorization.isPresent()) {
                        this.samplingCovariance = factorization.get().lMatrix();
                        this.determinant = factorization.get().determinant();
                        this.precision = factorization.get().inverse();
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

                double tmp = 1;
                for (int i = 0; i < this.covarianceVector.size(); i++) {
                    tmp *= this.covarianceVector.get(i);
                }
                this.determinant = tmp;
                this.precision = this.covarianceVector.copy();
                this.precision.foreachInPlace(a -> 1.0/a);

                // set unused variables.
                this.covarianceMatrix = null;
                this.samplingCovariance = null;
                this.variance = Double.NaN;
            }
            case SPHERICAL -> {
                if (covariance instanceof DenseVector vec) {
                    if ((vec.size() != 1) && (vec.size() != means.size())) {
                        throw new IllegalArgumentException("Covariance must be a single element vector for spherical covariance. Found " + vec.size());
                    } else if (vec.size() != 1) {
                        // check all elements are the same
                        double init = vec.get(0);
                        for (int i = 1; i < vec.size(); i++) {
                            if (init != vec.get(i)) {
                                throw new IllegalArgumentException("Covariance values must be the same for spherical covariance, found " + init + " at position 0 and " + vec.get(i) + " at position " + i);
                            }
                        }
                    }
                } else {
                    throw new IllegalArgumentException("Covariance must be a single element vector for spherical covariance, found " + covariance.getClass());
                }
                this.variance = vec.get(0);

                double tmp = 1;
                for (int i = 0; i < means.size(); i++) {
                    tmp *= this.variance;
                }
                this.determinant = tmp;
                this.precision = new DenseVector(1, 1.0 / variance);

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
    @Override
    public DenseVector sampleVector() {
        return sampleVector(rng);
    }

    /**
     * Sample a vector from this multivariate normal distribution.
     * @return A sample from this distribution.
     */
    @Override
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

        sampled.intersectAndAddInPlace(means);

        return sampled;
    }

    /**
     * Gets a copy of the mean vector.
     * @return A copy of the mean vector.
     */
    public DenseVector means() {
        return means.copy();
    }

    /**
     * Gets a copy of the covariance, either a {@link DenseMatrix} if it's full rank,
     * or a {@link DenseVector} if it's diagonal or spherical.
     * @return The covariance.
     */
    public Tensor covariance() {
        return switch (type) {
            case FULL -> covarianceMatrix.copy();
            case DIAGONAL -> covarianceVector.copy();
            case SPHERICAL -> new DenseVector(means.size(), variance);
        };
    }

    /**
     * Compute the log probability of the input under this multivariate normal distribution.
     * @param input The input to compute.
     * @return The log probability.
     */
    public double logProbability(SGDVector input) {
        return logProbability(input, means, precision, determinant, type);
    }

    /**
     * Computes the log probability of the input under the supplied parameters for a multivariate
     * normal distribution.
     * <p>
     * The parameters must be valid otherwise it will throw runtime exceptions.
     * @param input The input point to compute the log probability of.
     * @param mean The mean of the multivariate normal distribution.
     * @param determinant The determinant of the precision matrix.
     * @param type The covariance type.
     * @param precision The precision matrix. If type is {@link CovarianceType#FULL} must be a
     *                  {@link DenseMatrix}, if {@link CovarianceType#DIAGONAL} or
     *                  {@link CovarianceType#SPHERICAL} must be a {@link DenseVector}. Spherical
     *                  covariances should have a single element dense vector.
     * @return The log probability of the input point under the supplied distribution parameters.
     */
    public static double logProbability(SGDVector input, DenseVector mean, Tensor precision, double determinant, CovarianceType type) {
        // p(input|mean, variance) = \frac{1}{(2\pi)^{d/2} determinant^{1/2}} e^{-1/2 * (input - mean)^T * precision * (input - mean)}
        // log p(i|mu,sigma) = - log ({2\pi}^{d/2}) - log (determinant^{1/2}) + (-1/2 * (i - mu)^T * precision * (i - mu))
        double scalar = (- (mean.size() / 2.0) * Math.log(2 * Math.PI)) - (Math.log(determinant) / 2.0);
        DenseVector diff = (DenseVector) input.subtract(mean);
        double distance = switch (type) {
            case FULL -> {
                DenseMatrix precMat = (DenseMatrix) precision;
                yield precMat.leftMultiply(diff).dot(diff);
            }
            case DIAGONAL -> {
                // diff^T * diagonal precision * diff
                // = diff.hadamard(precision).dot(diff)
                // = diff.square().dot(precision)
                DenseVector precVec = (DenseVector) precision;
                diff.foreachInPlace(a -> a * a);
                yield diff.dot(precVec);
            }
            case SPHERICAL -> {
                double precVal = ((DenseVector) precision).get(0);
                diff.foreachInPlace(a -> a * a);
                diff.scaleInPlace(precVal);
                yield diff.sum();
            }
        };
        return scalar - (0.5 * distance);
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

        /**
         * Convert enum value into enum instance, used for serialization.
         * <p>
         * Throws {@link IllegalArgumentException} if the enum value is out of range.
         * @param value The enum value.
         * @return The enum type.
         */
        public static CovarianceType fromValue(int value) {
            CovarianceType[] values = CovarianceType.values();
            for (CovarianceType t : values) {
                if (t.value == value) {
                    return t;
                }
            }
            // Failed to find the enum.
            throw new IllegalArgumentException("Invalid CovarianceType enum value, found " + value);
        }
    }
}
