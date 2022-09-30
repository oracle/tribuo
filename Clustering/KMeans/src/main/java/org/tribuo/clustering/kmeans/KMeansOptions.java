/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.clustering.kmeans;

import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.Trainer;
import org.tribuo.clustering.kmeans.KMeansTrainer.Initialisation;
import org.tribuo.math.distance.DistanceType;

import java.util.logging.Logger;

/**
 * OLCUT {@link Options} for the K-Means implementation.
 */
public class KMeansOptions implements Options {
    private static final Logger logger = Logger.getLogger(KMeansOptions.class.getName());

    /**
     * Iterations of the k-means algorithm. Defaults to 10.
     */
    @Option(longName = "kmeans-interations", usage = "Iterations of the k-means algorithm. Defaults to 10.")
    public int iterations = 10;
    /**
     * Number of centroids in K-Means. Defaults to 10.
     */
    @Option(longName = "kmeans-num-centroids", usage = "Number of centroids in K-Means. Defaults to 10.")
    public int centroids = 10;
    /**
     * Distance function in K-Means. Defaults to L2 (EUCLIDEAN).
     */
    @Option(longName = "kmeans-distance-type", usage = "The type of distance function to use for various distance calculations.")
    public DistanceType distType = DistanceType.L2;
    /**
     * Initialisation function in K-Means. Defaults to RANDOM.
     */
    @Option(longName = "kmeans-initialisation", usage = "Initialisation function in K-Means. Defaults to RANDOM.")
    public Initialisation initialisation = Initialisation.RANDOM;
    /**
     * Number of computation threads in K-Means. Defaults to 4.
     */
    @Option(longName = "kmeans-num-threads", usage = "Number of computation threads in K-Means. Defaults to 4.")
    public int numThreads = 4;
    @Option(longName = "kmeans-seed", usage = "Sets the random seed for K-Means.")
    private long seed = Trainer.DEFAULT_SEED;

    /**
     * Gets the configured KMeansTrainer using the options in this object.
     * @return A KMeansTrainer.
     */
    public KMeansTrainer getTrainer() {
        logger.info("Configuring K-Means Trainer");
        return new KMeansTrainer(centroids, iterations, distType.getDistance(), initialisation, numThreads, seed);
    }
}
