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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class MultivariateNormalDistributionTest {

    @Test
    public void testMeanAndVarChol() {
        double[] mean = new double[]{2.5, 2.5};

        double[][] covariance = new double[][]{{1.0, 0.5}, {0.5, 1.0}};
        MultivariateNormalDistribution rng = new MultivariateNormalDistribution(mean, covariance, 12345);

        meanVarComp(rng,mean,covariance,1e-2);
    }

    @Test
    public void testMeanAndVarEigen() {
        double[] mean = new double[]{2.5, 2.5};

        double[][] covariance = new double[][]{{1.0, 0.5}, {0.5, 1.0}};
        MultivariateNormalDistribution rng = new MultivariateNormalDistribution(mean, covariance, 12345, true);

        meanVarComp(rng,mean,covariance,1e-2);
    }

    private static void meanVarComp(MultivariateNormalDistribution rng, double[] mean, double[][]covariance, double tolerance) {
        double[][] samples = new double[100000][];
        double[] computedMean = new double[2];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = rng.sampleArray();
            computedMean[0] += samples[i][0];
            computedMean[1] += samples[i][1];
        }
        computedMean[0] /= samples.length;
        computedMean[1] /= samples.length;

        double[][] computedCovariance = new double[2][2];
        for (int i = 0; i < samples.length; i++) {
            computedCovariance[0][0] += (samples[i][0] - computedMean[0]) * (samples[i][0] - computedMean[0]);
            computedCovariance[0][1] += (samples[i][0] - computedMean[0]) * (samples[i][1] - computedMean[1]);
            computedCovariance[1][0] += (samples[i][1] - computedMean[1]) * (samples[i][0] - computedMean[0]);
            computedCovariance[1][1] += (samples[i][1] - computedMean[1]) * (samples[i][1] - computedMean[1]);
        }
        computedCovariance[0][0] /= samples.length-1;
        computedCovariance[0][1] /= samples.length-1;
        computedCovariance[1][0] /= samples.length-1;
        computedCovariance[1][1] /= samples.length-1;

        assertArrayEquals(mean,computedMean,1e-2);
        assertArrayEquals(covariance[0],computedCovariance[0],tolerance);
        assertArrayEquals(covariance[1],computedCovariance[1],tolerance);
    }

}
