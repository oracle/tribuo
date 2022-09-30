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
 * The built-in distance functions.
 */
public enum DistanceType {
    /**
     * L1 (or Manhattan) distance.
     */
    L1(new L1Distance()),
    /**
     * L2 (or Euclidean) distance.
     */
    L2(new L2Distance()),
    /**
     * Cosine similarity used as a distance measure.
     */
    COSINE(new CosineDistance());

    private final Distance dist;

    private DistanceType(Distance dist) {
        this.dist = dist;
    }

    /**
     * Gets the Distance object specified by this enum.
     * @return The distance object.
     */
    public Distance getDistance() {
        return dist;
    }

    /**
     * Calculates the distance between two vectors.
     *
     * @param vector1 A {@link SGDVector} representing a data point.
     * @param vector2 A {@link SGDVector} representing a second data point.
     * @return A double representing the distance between the two points.
     */
    public double computeDistance(SGDVector vector1, SGDVector vector2) {
        return dist.computeDistance(vector1, vector2);
    }
}
