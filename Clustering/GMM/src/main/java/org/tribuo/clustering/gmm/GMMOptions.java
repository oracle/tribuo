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

package org.tribuo.clustering.gmm;

import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.Trainer;
import org.tribuo.clustering.gmm.GMMTrainer.Initialisation;
import org.tribuo.math.distributions.MultivariateNormalDistribution.CovarianceType;
import org.tribuo.math.distributions.MultivariateNormalDistribution;

import java.util.logging.Logger;

/**
 * OLCUT {@link Options} for the GMM implementation.
 */
public class GMMOptions implements Options {
    private static final Logger logger = Logger.getLogger(GMMOptions.class.getName());

    /**
     * Iterations of the GMM algorithm. Defaults to 10.
     */
    @Option(longName = "gmm-interations", usage = "Iterations of the GMM algorithm. Defaults to 10.")
    public int iterations = 10;
    /**
     * Number of centroids/Gaussians in GMM. Defaults to 10.
     */
    @Option(longName = "gmm-num-centroids", usage = "Number of centroids in GMM. Defaults to 10.")
    public int centroids = 10;
    /**
     * The covariance type of the Gaussians.
     */
    @Option(charName = 'v', longName = "covariance-type", usage = "Set the covariance type.")
    public CovarianceType covarianceType = MultivariateNormalDistribution.CovarianceType.DIAGONAL;
    /**
     * Initialisation function in GMM. Defaults to RANDOM.
     */
    @Option(longName = "gmm-initialisation", usage = "Initialisation function in GMM. Defaults to RANDOM.")
    public Initialisation initialisation = GMMTrainer.Initialisation.RANDOM;
    /**
     * Convergence tolerance to terminate EM early.
     */
    @Option(longName = "gmm-tolerance", usage = "The convergence threshold.")
    public double tolerance = 1e-3f;
    /**
     * Number of computation threads in GMM. Defaults to 4.
     */
    @Option(longName = "gmm-num-threads", usage = "Number of computation threads in GMM. Defaults to 4.")
    public int numThreads = 4;
    /**
     * The RNG seed.
     */
    @Option(longName = "gmm-seed", usage = "Sets the random seed for GMM.")
    public long seed = Trainer.DEFAULT_SEED;

    /**
     * Gets the configured GMMTrainer using the options in this object.
     * @return A GMMTrainer.
     */
    public GMMTrainer getTrainer() {
        logger.info("Configuring GMM Trainer");
        return new GMMTrainer(centroids, iterations, covarianceType, initialisation, tolerance, numThreads, seed);
    }
}
