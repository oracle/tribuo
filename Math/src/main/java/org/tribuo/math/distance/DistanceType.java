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

package org.tribuo.math.distance;

import org.tribuo.math.la.SGDVector;

/**
 * The available distance functions.
 */
public enum DistanceType {
    /**
     * L1 (or Manhattan) distance.
     */
    L1,
    /**
     * L2 (or Euclidean) distance.
     */
    L2,
    /**
     * Cosine similarity used as a distance measure.
     */
    COSINE;

    /**
     * Calculates the distance between two vectors.
     *
     * @param vector1 A {@link SGDVector} representing a data point.
     * @param vector2 A {@link SGDVector} representing a second data point.
     * @param distanceType The {@link DistanceType} function to employ.
     * @return A double representing the distance between the two points.
     */
    public static double getDistance(SGDVector vector1, SGDVector vector2, DistanceType distanceType) {
        double distance;
        switch (distanceType) {
            case L1:
                distance = vector1.l1Distance(vector2);
                break;
            case L2:
                distance = vector1.l2Distance(vector2);
                break;
            case COSINE:
                distance = vector1.cosineDistance(vector2);
                break;
            default:
                throw new IllegalStateException("Unknown distance " + distanceType);
        }
        return distance;
    }
}
