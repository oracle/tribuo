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

package org.tribuo.clustering.hdbscan;

import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.clustering.hdbscan.HdbscanTrainer.Distance;

import java.util.logging.Logger;

/**
 * OLCUT {@link Options} for the HDBSCAN* implementation.
 */
public final class HdbscanOptions implements Options {
    private static final Logger logger = Logger.getLogger(HdbscanOptions.class.getName());
    @Override
    public String getOptionsDescription() {
        return "Options for configuring a HdbscanTrainer.";
    }

    /**
     * The minimum number of points required to form a cluster. Defaults to 5.
     */
    @Option(longName = "minimum-cluster-size", usage = "The minimum number of points required to form a cluster.")
    public int minClusterSize = 5;

    /**
     * Distance function in HDBSCAN*. Defaults to EUCLIDEAN.
     */
    @Option(longName = "distance-function", usage = "Distance function to use for various distance calculations.")
    public Distance distanceType = Distance.EUCLIDEAN;

    /**
     * The number of nearest-neighbors to use in the initial density approximation. Defaults to 5.
     */
    @Option(longName = "k-nearest-neighbors", usage = "The number of nearest-neighbors to use in the initial density approximation. " +
        "The value includes the point itself.")
    public int k = 5;

    /**
     * Number of threads to use for training the hdbscan model. Defaults to 2.
     */
    @Option(longName = "hdbscan-num-threads", usage = "Number of threads to use for training the hdbscan model.")
    public int numThreads = 2;

    /**
     * Gets the configured HdbscanTrainer using the options in this object.
     * @return A HdbscanTrainer.
     */
    public HdbscanTrainer getTrainer() {
        logger.info("Configuring Hdbscan Trainer");
        return new HdbscanTrainer(minClusterSize, distanceType, k, numThreads);
    }
}
