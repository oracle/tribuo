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
import java.util.List;

/**
 * A data source of two interleaved half circles with some zero mean Gaussian noise applied to each point.
 */
public final class NoisyInterlockingCrescentsDataSource extends DemoLabelDataSource {

    @Config(description = "Variance of the Gaussian noise")
    private double variance = 0.1;

    /**
     * For OLCUT.
     */
    private NoisyInterlockingCrescentsDataSource() {
        super();
    }

    /**
     * Constructs a noisy interlocking crescents data source.
     * <p>
     * It's the same as {@link InterlockingCrescentsDataSource} but each point has Gaussian
     * noise with zero mean and the specified variance added to it.
     *
     * @param numSamples The number of samples to generate.
     * @param seed       The RNG seed.
     * @param variance   The variance of the Gaussian noise.
     */
    public NoisyInterlockingCrescentsDataSource(int numSamples, long seed, double variance) {
        super(numSamples, seed);
        this.variance = variance;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (variance <= 0.0) {
            throw new PropertyException("", "variance", "Variance must be positive, found " + variance);
        }
        super.postConfig();
    }

    @Override
    protected List<Example<Label>> generate() {
        List<Example<Label>> list = new ArrayList<>();

        for (int i = 0; i < numSamples / 2; i++) {
            double[] values = new double[2];
            double u = rng.nextDouble();
            values[0] = Math.cos(Math.PI * u) + rng.nextGaussian() * variance;
            values[1] = Math.sin(Math.PI * u) + rng.nextGaussian() * variance;
            list.add(new ArrayExample<>(FIRST_CLASS, FEATURE_NAMES, values));
        }

        for (int i = numSamples / 2; i < numSamples; i++) {
            double[] values = new double[2];
            double u = rng.nextDouble();
            values[0] = (1 - Math.cos(Math.PI * u)) + rng.nextGaussian() * variance;
            values[1] = (0.5 - Math.sin(Math.PI * u)) + rng.nextGaussian() * variance;
            list.add(new ArrayExample<>(SECOND_CLASS, FEATURE_NAMES, values));
        }

        return list;
    }

    @Override
    public String toString() {
        return "NoisyInterlockingCrescents(numSamples=" + numSamples + ",seed=" + seed + ",variance=" + variance + ')';
    }
}
