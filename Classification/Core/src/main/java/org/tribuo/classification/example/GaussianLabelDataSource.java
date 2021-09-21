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

package org.tribuo.classification.example;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.tribuo.Example;
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * A data source for two classes generated from separate Gaussians.
 */
public final class GaussianLabelDataSource extends DemoLabelDataSource {

    @Config(mandatory = true, description = "2d mean of the first Gaussian.")
    private double[] firstMean;

    @Config(mandatory = true, description = "4 element covariance matrix of the first Gaussian.")
    private double[] firstCovarianceMatrix;

    @Config(mandatory = true, description = "2d mean of the second Gaussian.")
    private double[] secondMean;

    @Config(mandatory = true, description = "4 element covariance matrix of the second Gaussian.")
    private double[] secondCovarianceMatrix;

    private double[] firstCholesky;

    private double[] secondCholesky;

    /**
     * For OLCUT.
     */
    private GaussianLabelDataSource() {
        super();
    }

    /**
     * Constructs a data source which contains two classes where each class is sampled from a 2d Gaussian with
     * the specified parameters.
     *
     * @param numSamples             The number of samples to draw.
     * @param seed                   The RNG seed.
     * @param firstMean              The mean of class one's Gaussian.
     * @param firstCovarianceMatrix  The covariance matrix of class one's Gaussian.
     * @param secondMean             The mean of class two's Gaussian.
     * @param secondCovarianceMatrix The covariance matrix of class two's Gaussian.
     */
    public GaussianLabelDataSource(int numSamples, long seed, double[] firstMean, double[] firstCovarianceMatrix, double[] secondMean, double[] secondCovarianceMatrix) {
        super(numSamples, seed);
        this.firstMean = firstMean;
        this.firstCovarianceMatrix = firstCovarianceMatrix;
        this.secondMean = secondMean;
        this.secondCovarianceMatrix = secondCovarianceMatrix;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (firstMean.length != 2) {
            throw new PropertyException("", "firstMean", "firstMean is not the right length");
        }
        if (secondMean.length != 2) {
            throw new PropertyException("", "secondMean", "secondMean is not the right length");
        }
        if (firstCovarianceMatrix.length != 4) {
            throw new PropertyException("", "firstCovarianceMatrix", "firstCovarianceMatrix is not the right length");
        }
        if (secondCovarianceMatrix.length != 4) {
            throw new PropertyException("", "secondCovarianceMatrix", "secondCovarianceMatrix is not the right length");
        }

        for (int i = 0; i < firstCovarianceMatrix.length; i++) {
            if (firstCovarianceMatrix[i] < 0) {
                throw new PropertyException("", "firstCovarianceMatrix", "First covariance matrix is not positive semi-definite");
            }
            if (secondCovarianceMatrix[i] < 0) {
                throw new PropertyException("", "secondCovarianceMatrix", "Second covariance matrix is not positive semi-definite");
            }
        }

        if (firstCovarianceMatrix[1] != firstCovarianceMatrix[2]) {
            throw new PropertyException("", "firstCovarianceMatrix", "First covariance matrix is not a covariance matrix");
        }

        if (secondCovarianceMatrix[1] != secondCovarianceMatrix[2]) {
            throw new PropertyException("", "secondCovarianceMatrix", "Second covariance matrix is not a covariance matrix");
        }

        firstCholesky = new double[3];
        firstCholesky[0] = Math.sqrt(firstCovarianceMatrix[0]);
        firstCholesky[1] = firstCovarianceMatrix[1] / Math.sqrt(firstCovarianceMatrix[0]);
        firstCholesky[2] = Math.sqrt(firstCovarianceMatrix[3] * firstCovarianceMatrix[0] - firstCovarianceMatrix[1] * firstCovarianceMatrix[1]) / Math.sqrt(firstCovarianceMatrix[0]);

        secondCholesky = new double[3];
        secondCholesky[0] = Math.sqrt(secondCovarianceMatrix[0]);
        secondCholesky[1] = secondCovarianceMatrix[1] / Math.sqrt(secondCovarianceMatrix[0]);
        secondCholesky[2] = Math.sqrt(secondCovarianceMatrix[3] * secondCovarianceMatrix[0] - secondCovarianceMatrix[1] * secondCovarianceMatrix[1]) / Math.sqrt(secondCovarianceMatrix[0]);
        super.postConfig();
    }

    @Override
    protected List<Example<Label>> generate() {
        List<Example<Label>> list = new ArrayList<>();

        for (int i = 0; i < numSamples / 2; i++) {
            double[] sample = sampleGaussian(rng, firstMean, firstCholesky);
            ArrayExample<Label> datapoint = new ArrayExample<>(FIRST_CLASS, FEATURE_NAMES, sample);
            list.add(datapoint);
        }

        for (int i = numSamples / 2; i < numSamples; i++) {
            double[] sample = sampleGaussian(rng, secondMean, secondCholesky);
            ArrayExample<Label> datapoint = new ArrayExample<>(SECOND_CLASS, FEATURE_NAMES, sample);
            list.add(datapoint);
        }

        return list;
    }

    /**
     * Samples from a 2d Gaussian specified by the mean vector and the Cholesky factorization.
     *
     * @param rng      The RNG to use.
     * @param means    The mean of the Gaussian.
     * @param cholesky The Cholesky factorization.
     * @return A sample from a 2d Gaussian.
     */
    private static double[] sampleGaussian(Random rng, double[] means, double[] cholesky) {
        double[] sample = new double[2];

        double first = rng.nextGaussian();
        sample[0] = means[0] + first * cholesky[0];
        double second = rng.nextGaussian();
        sample[1] = means[1] + (first * cholesky[1]) + (second * cholesky[2]);

        return sample;
    }

    @Override
    public String toString() {
        String sb = "GaussianGenerator(numSamples=" +
                numSamples +
                ",seed=" +
                seed +
                ",firstMean=" +
                Arrays.toString(firstMean) +
                ",firstCovarianceMatrix=" +
                Arrays.toString(firstCovarianceMatrix) +
                ",secondMean=" +
                Arrays.toString(secondMean) +
                ",secondCovarianceMatrix=" +
                Arrays.toString(secondCovarianceMatrix) +
                ')';

        return sb;
    }
}
