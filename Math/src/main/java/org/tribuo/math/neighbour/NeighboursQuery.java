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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.la.SGDVector;

import java.util.List;

/**
 * An interface for nearest neighbour query objects.
 */
public interface NeighboursQuery {

    /**
     * Queries a set of {@link SGDVector}s to determine the k points nearest to the provided point. When there are
     * multiple points equidistant from the provided point, the order in which they are returned may vary depending on
     * the implementation.
     * @param point The point to determine the nearest k points for.
     * @param k The number of neighbouring points to identify.
     * @return A list of k {@link Pair}s, where a pair contains the index of the neighbouring point in the original
     *         data and the distance between this point and the provided point.
     */
    List<Pair<Integer, Double>> query(SGDVector point, int k);

    /**
     * Queries a set of {@link SGDVector}s to determine the k points nearest to the provided points. When there are
     * multiple points equidistant from a provided point, the order in which they are returned may vary depending on
     * the implementation.
     * @param points An array of points to determine the nearest k points for.
     * @param k The number of neighbouring points to identify.
     * @return An list containing lists of k {@link Pair}s. There is list entry for each provided point which is a
     *         list of k pairs. Each pair contains the index of the neighbouring point in the original data and the
     *         distance between this point and the provided point.
     */
    List<List<Pair<Integer, Double>>> query(SGDVector[] points, int k);

    /**
     * Queries a set of {@link SGDVector}s to determine the k points nearest to every point in the set. When there are
     * multiple points equidistant from a point in the set, the order in which they are returned may vary depending on
     * the implementation.
     * @param k The number of neighbouring points to identify.
     * @return A list containing lists of k {@link Pair}s. There is list entry for each provided point which is a
     *         list of k pairs. Each pair contains the index of the neighbouring point in the original data and the
     *         distance between this point and the provided point.
     */
    List<List<Pair<Integer, Double>>> queryAll(int k);
}
