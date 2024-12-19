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

import java.util.random.RandomGenerator;

/**
 * Interface for probability distributions which can be sampled from.
 * <p>
 * The vector sampled represents a single sample from that (possibly multivariate)
 * distribution rather than a sequence of samples.
 */
public interface Distribution {

    /**
     * Sample a single vector from this probability distribution.
     * @return A vector sampled from the distribution.
     */
    DenseVector sampleVector();

    /**
     * Sample a single vector from this probability distribution using the supplied RNG.
     * @param otherRNG The RNG to use.
     * @return A vector sampled from this distribution.
     */
    DenseVector sampleVector(RandomGenerator otherRNG);

    /**
     * Sample a vector from this probability distribution and return it as an array.
     * @return An array sampled from this distribution.
     */
    default double[] sampleArray() {
        return sampleVector().toArray();
    }
}
