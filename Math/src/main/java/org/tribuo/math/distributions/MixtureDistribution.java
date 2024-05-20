/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.math.la.DenseVector;
import org.tribuo.util.Util;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A mixture distribution which samples from a set of internal distributions mixed by some probability distribution.
 * @param <T> The inner distribution type.
 */
public final class MixtureDistribution<T extends Distribution> implements Distribution {

    private final List<T> dists;

    private final double[] mixingDistribution;

    private final double[] cdf;

    private final RandomGenerator rng;

    private final long seed;

    /**
     * Construct a mixture distribution over the supplied components.
     * @param distributions The distribution components.
     * @param mixingDistribution The mixing distribution, must be a valid PMF.
     * @param seed The RNG seed.
     */
    public MixtureDistribution(List<T> distributions, DenseVector mixingDistribution, long seed) {
        this(distributions, mixingDistribution.toArray(), seed);
    }

    /**
     * Construct a mixture distribution over the supplied components.
     * @param distributions The distribution components.
     * @param mixingDistribution The mixing distribution, must be a valid PMF.
     * @param seed The RNG seed.
     */
    public MixtureDistribution(List<T> distributions, double[] mixingDistribution, long seed) {
        this.dists = List.copyOf(distributions);
        this.mixingDistribution = Arrays.copyOf(mixingDistribution, mixingDistribution.length);
        this.seed = seed;
        this.rng = new SplittableRandom(seed);
        if (dists.size() != this.mixingDistribution.length) {
            throw new IllegalArgumentException("Invalid distribution, expected the same number of components as probabilities, found " + dists.size() + " components, and " + this.mixingDistribution.length + " probabilities");
        }
        if (!Util.validatePMF(this.mixingDistribution)) {
            throw new IllegalArgumentException("Invalid mixing distribution, was not a valid PMF, found " + Arrays.toString(this.mixingDistribution));
        }
        this.cdf = Util.generateCDF(this.mixingDistribution);
    }

    /**
     * Returns the number of distributions.
     * @return The number of distributions.
     */
    public int getNumComponents() {
        return dists.size();
    }

    /**
     * Return a mixture component.
     * @param i The index of the mixture component.
     * @return The ith component.
     */
    public T getComponent(int i) {
        return dists.get(i);
    }

    /**
     * Returns a copy of the mixing distribution.
     * @return A copy of the mixing distribution.
     */
    public double[] getMixingDistribution() {
        return Arrays.copyOf(mixingDistribution, mixingDistribution.length);
    }

    @Override
    public DenseVector sampleVector() {
        return sampleVector(rng);
    }

    @Override
    public DenseVector sampleVector(RandomGenerator otherRNG) {
        int idx = Util.sampleFromCDF(cdf, otherRNG);
        return dists.get(idx).sampleVector();
    }

    @Override
    public String toString() {
        return "Mixture(seed="+seed+",mixingDistribution="+ Arrays.toString(mixingDistribution) +",components="+dists+")";
    }
}
