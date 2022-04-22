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

package org.tribuo.math.neighbour.kdtree;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.neighbour.NeighboursQueryFactory;

/**
 * A factory which creates k-d tree nearest neighbour query objects.
 */
public class KDTreeFactory implements NeighboursQueryFactory {
    private static final long serialVersionUID = 1L;

    @Config(description = "The distance function to use.")
    private DistanceType distanceType = DistanceType.L2;

    @Config(description = "The number of threads to use for training.")
    private int numThreads = 1;

    /**
     * for olcut.
     */
    private KDTreeFactory() {
    }

    /**
     * Constructs a k-d tree nearest neighbor query factory object using the supplied parameters.
     * @param distanceType The distance function.
     * @param numThreads The number of threads to be used to parallelize the computation.
     */
    public KDTreeFactory(DistanceType distanceType, int numThreads) {
        this.distanceType = distanceType;
        this.numThreads = numThreads;
        postConfig();
    }

    /**
     * Constructs a k-d tree nearest neighbor query object using the supplied array of {@link SGDVector}.
     * @param data An array of {@link SGDVector}.
     */
    @Override
    public KDTree createNeighboursQuery(SGDVector[] data) {
        return new KDTree(data, this.distanceType, this.numThreads);
    }

    @Override
    public DistanceType getDistanceType() {
        return distanceType;
    }

    @Override
    public int getNumThreads() {
        return numThreads;
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public synchronized void postConfig() {
        if (numThreads <= 0) {
            throw new PropertyException("numThreads", "The number of threads must be a number greater than 0.");
        }
    }
}
