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

package org.tribuo.math.neighbour;

import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForceFactory;
import org.tribuo.math.neighbour.kdtree.KDTreeFactory;

/**
 * These are the supported neighbour query implementations. A factory object is used to emit instances of its
 * implementation.
 */
public enum NeighboursQueryFactoryType {

    BRUTE_FORCE,

    KD_TREE;

    /**
     * Returns the appropriate {@link NeighboursQueryFactory}.
     * @param distanceType The distance function.
     * @param numThreads The number of threads to be used to parallelize the computation.
     */
    public static NeighboursQueryFactory getNeighboursQueryFactory(NeighboursQueryFactoryType nqFactoryType,
                                                                   DistanceType distanceType, int numThreads) {
        NeighboursQueryFactory neighboursQueryFactory;
        switch(nqFactoryType) {
            case BRUTE_FORCE:
                neighboursQueryFactory = new NeighboursBruteForceFactory(distanceType, numThreads);
                break;
            case KD_TREE:
                neighboursQueryFactory = new KDTreeFactory(distanceType, numThreads);
                break;
            default:
                throw new IllegalStateException("Unknown neighbour query factory " + nqFactoryType);
        }
        return neighboursQueryFactory;
    }
}


