/*
 * Copyright (c) 2021, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.clustering.example;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.OutputFactory;
import org.tribuo.Trainer;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ClusteringFactory;
import org.tribuo.impl.ArrayExample;
import org.tribuo.math.distributions.MultivariateNormalDistribution;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.util.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Generates a clustering dataset drawn from a mixture of 5 Gaussians.
 * <p>
 * The Gaussians can be at most 4 dimensional, resulting in 4 features.
 * <p>
 * By default the Gaussians are 2-dimensional with the following means and variances:
 * <ul>
 *     <li>N([0.0,0.0], [[1.0,0.0],[0.0,1.0]])</li>
 *     <li>N([5.0,5.0], [[1.0,0.0],[0.0,1.0]])</li>
 *     <li>N([2.5,2.5], [[1.0,0.5],[0.5,1.0]])</li>
 *     <li>N([10.0,0.0], [[0.1,0.0],[0.0,0.1]])</li>
 *     <li>N([-1.0,0.0], [[1.0,0.0],[0.0,0.1]])</li>
 * </ul>
 * and the mixing distribution is:
 * [0.1, 0.35, 0.05, 0.25, 0.25].
 */
public final class GaussianClusterDataSource implements ConfigurableDataSource<ClusterID> {

    private static final ClusteringFactory factory = new ClusteringFactory();

    private static final String[] allFeatureNames = new String[]{
            "A", "B", "C", "D",
    };

    @Config(mandatory = true, description = "The number of samples to draw.")
    private int numSamples;

    @Config(description = "The probability of sampling from each Gaussian, must sum to 1.0.")
    private double[] mixingDistribution = new double[]{0.1, 0.35, 0.05, 0.25, 0.25};

    @Config(description = "The mean of the first Gaussian.")
    private double[] firstMean = new double[]{0.0, 0.0};

    @Config(description = "A vector representing the first Gaussian's covariance matrix.")
    private double[] firstVariance = new double[]{1.0, 0.0, 0.0, 1.0};

    @Config(description = "The mean of the second Gaussian.")
    private double[] secondMean = new double[]{5.0, 5.0};

    @Config(description = "A vector representing the second Gaussian's covariance matrix.")
    private double[] secondVariance = new double[]{1.0, 0.0, 0.0, 1.0};

    @Config(description = "The mean of the third Gaussian.")
    private double[] thirdMean = new double[]{2.5, 2.5};

    @Config(description = "A vector representing the third Gaussian's covariance matrix.")
    private double[] thirdVariance = new double[]{1.0, 0.5, 0.5, 1.0};

    @Config(description = "The mean of the fourth Gaussian.")
    private double[] fourthMean = new double[]{10.0, 0.0};

    @Config(description = "A vector representing the fourth Gaussian's covariance matrix.")
    private double[] fourthVariance = new double[]{0.1, 0.0, 0.0, 0.1};

    @Config(description = "The mean of the fifth Gaussian.")
    private double[] fifthMean = new double[]{-1.0, 0.0};

    @Config(description = "A vector representing the fifth Gaussian's covariance matrix.")
    private double[] fifthVariance = new double[]{1.0, 0.0, 0.0, 0.1};

    @Config(description = "The RNG seed.")
    private long seed = Trainer.DEFAULT_SEED;

    private List<Example<ClusterID>> examples;

    /**
     * For OLCUT.
     */
    private GaussianClusterDataSource() {
    }

    /**
     * Generates a clustering dataset drawn from a mixture of 5 Gaussians.
     * <p>
     * The default Gaussians are:
     * <ul>
     *     <li>N([0.0,0.0], [[1.0,0.0],[0.0,1.0]])</li>
     *     <li>N([5.0,5.0], [[1.0,0.0],[0.0,1.0]])</li>
     *     <li>N([2.5,2.5], [[1.0,0.5],[0.5,1.0]])</li>
     *     <li>N([10.0,0.0], [[0.1,0.0],[0.0,0.1]])</li>
     *     <li>N([-1.0,0.0], [[1.0,0.0],[0.0,0.1]])</li>
     * </ul>
     * and the mixing distribution is:
     * [0.1, 0.35, 0.05, 0.25, 0.25].
     *
     * @param numSamples The size of the output dataset.
     * @param seed       The rng seed to use.
     */
    public GaussianClusterDataSource(int numSamples, long seed) {
        this.numSamples = numSamples;
        this.seed = seed;
        postConfig();
    }

    /**
     * Generates a clustering dataset drawn from a mixture of 5 Gaussians.
     * <p>
     * The Gaussians can be at most 4 dimensional, resulting in 4 features.
     *
     * @param numSamples         The size of the output dataset.
     * @param mixingDistribution The probability of each cluster.
     * @param firstMean          The mean of the first Gaussian.
     * @param firstVariance      The variance of the first Gaussian, linearised from a row-major matrix.
     * @param secondMean         The mean of the second Gaussian.
     * @param secondVariance     The variance of the second Gaussian, linearised from a row-major matrix.
     * @param thirdMean          The mean of the third Gaussian.
     * @param thirdVariance      The variance of the third Gaussian, linearised from a row-major matrix.
     * @param fourthMean         The mean of the fourth Gaussian.
     * @param fourthVariance     The variance of the fourth Gaussian, linearised from a row-major matrix.
     * @param fifthMean          The mean of the fifth Gaussian.
     * @param fifthVariance      The variance of the fifth Gaussian, linearised from a row-major matrix.
     * @param seed               The rng seed to use.
     */
    public GaussianClusterDataSource(int numSamples, double[] mixingDistribution,
                                     double[] firstMean, double[] firstVariance,
                                     double[] secondMean, double[] secondVariance,
                                     double[] thirdMean, double[] thirdVariance,
                                     double[] fourthMean, double[] fourthVariance,
                                     double[] fifthMean, double[] fifthVariance,
                                     long seed) {
        this.numSamples = numSamples;
        this.mixingDistribution = mixingDistribution;
        this.firstMean = firstMean;
        this.firstVariance = firstVariance;
        this.secondMean = secondMean;
        this.secondVariance = secondVariance;
        this.thirdMean = thirdMean;
        this.thirdVariance = thirdVariance;
        this.fourthMean = fourthMean;
        this.fourthVariance = fourthVariance;
        this.fifthMean = fifthMean;
        this.fifthVariance = fifthVariance;
        this.seed = seed;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (numSamples < 1) {
            throw new PropertyException("", "numSamples", "numSamples must be positive, found " + numSamples);
        }
        if (mixingDistribution.length != 5) {
            throw new PropertyException("", "mixingDistribution", "mixingDistribution must have 5 elements, found " + mixingDistribution.length);
        }
        if (Math.abs(Util.sum(mixingDistribution) - 1.0) > 1e-10) {
            throw new PropertyException("", "mixingDistribution", "mixingDistribution must sum to 1.0, found " + Util.sum(mixingDistribution));
        }
        if ((firstMean.length > allFeatureNames.length) || (firstMean.length == 0)) {
            throw new PropertyException("", "firstMean", "Must have 1-4 features, found " + firstMean.length);
        }
        int covarianceSize = firstMean.length * firstMean.length;
        if (firstVariance.length != (covarianceSize)) {
            throw new PropertyException("", "firstVariance", "Invalid first covariance matrix, expected " + covarianceSize + " elements, found " + firstVariance.length);
        }
        if (secondMean.length != firstMean.length) {
            throw new PropertyException("", "secondMean", "All Gaussians must have the same number of dimensions, expected " + firstMean.length + ", found " + secondMean.length);
        }
        if (secondVariance.length != firstVariance.length) {
            throw new PropertyException("", "secondVariance", "secondVariance is invalid, expected " + covarianceSize + ", found " + secondVariance.length);
        }
        if (thirdMean.length != firstMean.length) {
            throw new PropertyException("", "thirdMean", "All Gaussians must have the same number of dimensions, expected " + firstMean.length + ", found " + thirdMean.length);
        }
        if (thirdVariance.length != firstVariance.length) {
            throw new PropertyException("", "thirdVariance", "thirdVariance is invalid, expected " + covarianceSize + ", found " + thirdVariance.length);
        }
        if (fourthMean.length != firstMean.length) {
            throw new PropertyException("", "fourthMean", "All Gaussians must have the same number of dimensions, expected " + firstMean.length + ", found " + fourthMean.length);
        }
        if (fourthVariance.length != firstVariance.length) {
            throw new PropertyException("", "fourthVariance", "fourthVariance is invalid, expected " + covarianceSize + ", found " + fourthVariance.length);
        }
        if (fifthMean.length != firstMean.length) {
            throw new PropertyException("", "fifthMean", "All Gaussians must have the same number of dimensions, expected " + firstMean.length + ", found " + fifthMean.length);
        }
        if (fifthVariance.length != firstVariance.length) {
            throw new PropertyException("", "fifthVariance", "fifthVariance is invalid, expected " + covarianceSize + ", found " + fifthVariance.length);
        }
        for (int i = 0; i < mixingDistribution.length; i++) {
            if (mixingDistribution[i] < 0) {
                throw new PropertyException("", "mixingDistribution", "Probability values in the mixing distribution must be non-negative, found " + Arrays.toString(mixingDistribution));
            }
        }
        double[] mixingCDF = Util.generateCDF(mixingDistribution);
        String[] featureNames = Arrays.copyOf(allFeatureNames, firstMean.length);
        Random rng = new Random(seed);
        MultivariateNormalDistribution first = new MultivariateNormalDistribution(
                firstMean, reshapeAndValidate(firstVariance, "firstVariance"), rng.nextInt(), true
        );
        MultivariateNormalDistribution second = new MultivariateNormalDistribution(
                secondMean, reshapeAndValidate(secondVariance, "secondVariance"), rng.nextInt(), true
        );
        MultivariateNormalDistribution third = new MultivariateNormalDistribution(
                thirdMean, reshapeAndValidate(thirdVariance, "thirdVariance"), rng.nextInt(), true
        );
        MultivariateNormalDistribution fourth = new MultivariateNormalDistribution(
                fourthMean, reshapeAndValidate(fourthVariance, "fourthVariance"), rng.nextInt(), true
        );
        MultivariateNormalDistribution fifth = new MultivariateNormalDistribution(
                fifthMean, reshapeAndValidate(fifthVariance, "fifthVariance"), rng.nextInt(), true
        );
        MultivariateNormalDistribution[] Gaussians = new MultivariateNormalDistribution[]{first, second, third, fourth, fifth};
        List<Example<ClusterID>> examples = new ArrayList<>(numSamples);
        for (int i = 0; i < numSamples; i++) {
            int centroid = Util.sampleFromCDF(mixingCDF, rng);
            double[] sample = Gaussians[centroid].sampleArray();
            examples.add(new ArrayExample<>(new ClusterID(centroid), featureNames, sample));
        }
        this.examples = Collections.unmodifiableList(examples);
    }

    @Override
    public OutputFactory<ClusterID> getOutputFactory() {
        return factory;
    }

    @Override
    public DataSourceProvenance getProvenance() {
        return new GaussianClusterDataSourceProvenance(this);
    }

    @Override
    public Iterator<Example<ClusterID>> iterator() {
        return examples.iterator();
    }

    /**
     * Reshapes the vector into a covariance matrix, validating that it's non-negative.
     *
     * @param vector The vector.
     * @return The matrix assuming the vector is linearised in row-major order.
     */
    private static double[][] reshapeAndValidate(double[] vector, String fieldName) {
        int length = (int) Math.sqrt(vector.length);
        if (length * length != vector.length) {
            throw new IllegalArgumentException("The vector does not represent a square matrix, found " + vector.length + " elements, which is not square.");
        }
        double[][] matrix = new double[length][length];
        for (int i = 0; i < vector.length; i++) {
            if (vector[i] < 0) {
                throw new PropertyException("", fieldName, fieldName + " must have a non-negative covariance matrix, found " + Arrays.toString(vector));
            }
            matrix[i / length][i % length] = vector[i];
        }
        return matrix;
    }

    /**
     * Generates a clustering dataset drawn from a mixture of 5 Gaussians.
     * <p>
     * The Gaussians can be at most 4 dimensional, resulting in 4 features.
     *
     * @param numSamples         The size of the output dataset.
     * @param mixingDistribution The probability of each cluster.
     * @param firstMean          The mean of the first Gaussian.
     * @param firstVariance      The variance of the first Gaussian, linearised from a row-major matrix.
     * @param secondMean         The mean of the second Gaussian.
     * @param secondVariance     The variance of the second Gaussian, linearised from a row-major matrix.
     * @param thirdMean          The mean of the third Gaussian.
     * @param thirdVariance      The variance of the third Gaussian, linearised from a row-major matrix.
     * @param fourthMean         The mean of the fourth Gaussian.
     * @param fourthVariance     The variance of the fourth Gaussian, linearised from a row-major matrix.
     * @param fifthMean          The mean of the fifth Gaussian.
     * @param fifthVariance      The variance of the fifth Gaussian, linearised from a row-major matrix.
     * @param seed               The rng seed to use.
     * @return A dataset drawn from a mixture of Gaussians.
     */
    public static MutableDataset<ClusterID> generateDataset(int numSamples, double[] mixingDistribution,
                                                     double[] firstMean, double[] firstVariance,
                                                     double[] secondMean, double[] secondVariance,
                                                     double[] thirdMean, double[] thirdVariance,
                                                     double[] fourthMean, double[] fourthVariance,
                                                     double[] fifthMean, double[] fifthVariance,
                                                     long seed) {
        GaussianClusterDataSource source = new GaussianClusterDataSource(numSamples, mixingDistribution,
                firstMean, firstVariance, secondMean, secondVariance, thirdMean, thirdVariance, fourthMean, fourthVariance,
                fifthMean, fifthVariance, seed);
        return new MutableDataset<>(source);
    }

    /**
     * Provenance for {@link GaussianClusterDataSource}.
     */
    public static final class GaussianClusterDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * Constructs a provenance from the host data source.
         *
         * @param host The host to read.
         */
        GaussianClusterDataSourceProvenance(GaussianClusterDataSource host) {
            super(host, "DataSource");
        }

        /**
         * Constructs a provenance from the marshalled form.
         *
         * @param map The map of field values.
         */
        public GaussianClusterDataSourceProvenance(Map<String, Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private GaussianClusterDataSourceProvenance(ExtractedInfo info) {
            super(info);
        }

        /**
         * Extracts the relevant provenance information fields for this class.
         *
         * @param map The map to remove values from.
         * @return The extracted information.
         */
        protected static ExtractedInfo extractProvenanceInfo(Map<String, Provenance> map) {
            Map<String, Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters, CLASS_NAME, StringProvenance.class, GaussianClusterDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters, HOST_SHORT_NAME, StringProvenance.class, GaussianClusterDataSourceProvenance.class.getSimpleName()).getValue();

            return new ExtractedInfo(className, hostTypeStringName, configuredParameters, Collections.emptyMap());
        }
    }
}
