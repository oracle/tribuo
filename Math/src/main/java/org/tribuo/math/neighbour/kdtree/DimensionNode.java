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

import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.SGDVector;

import static org.tribuo.math.neighbour.kdtree.KDTree.IntAndVector;
import static org.tribuo.math.neighbour.kdtree.KDTree.DistanceRecordBoundedMinHeap;

/**
 * A node used to in a k-d tree ({@link KDTree}). A node is a point from a dataset and is placed according to the value
 * at a specific dimension of the point.
 */
public class DimensionNode {

    private final int dimension;

    private final IntAndVector record;

    private final int maxD;

    private final double coord;

    private final DistanceType distanceType;

    private DimensionNode below;

    private DimensionNode above;

    /**
     * Constructs a dimension node using a record containing a {@link SGDVector} and its original index position and
     * the dimension of the point. The default region has no bounds for all dimensions.
     *
     * @param dimension The dimension that this node represents.
     * @param record The point.
     * @param distanceType The distance function.
     */
    public DimensionNode(int dimension, IntAndVector record, DistanceType distanceType) {
        this.dimension = dimension;
        this.record = record;
        this.maxD = record.vector.size();
        this.distanceType = distanceType;
        this.coord = record.vector.get(dimension-1);
    }

    /**
     * Return the node below this one in the k-d tree.
     * @return The node below this one.
     */
    public DimensionNode getBelow() {
        return below;
    }

    /**
     * Return the node above this one in the k-d tree.
     * @return The node below this one.
     */
    public DimensionNode getAbove() {
        return above;
    }

    /**
     * Set the node below this one in the k-d tree.
     *
     * @param node The node being added to the tree.
     */
    public void setBelow(DimensionNode node) {
        if (node == null) {
            this.below = null;
        }
        // Ensure the dimension of node being added is only 1 greater, or is the dimension wrap around case.
        else if ((this.dimension + 1 == node.dimension) || (this.dimension == maxD && node.dimension == 1)) {
            this.below = node;
        }
        else {
            throw new IllegalArgumentException("Setting the below/left node failed because the dimensions are incorrect.");
        }
    }

    /**
     * Set the node above this one in the k-d tree.
     *
     * @param node The node being added to the tree.
     */
    public void setAbove(DimensionNode node) {
        if (node == null) {
            this.above = null;
        }
        // Ensure the dimension of node being added is only 1 greater, or is the dimension wrap around case.
        else if ((this.dimension + 1 == node.dimension) || (this.dimension == maxD && node.dimension == 1)) {
            this.above = node;
        }
        else {
            throw new IllegalArgumentException("Setting the above/right node failed because the dimensions are incorrect.");
        }
    }

    /**
     * Tests if the provided point is strictly below the line this node and its dimension represent.
     *
     * @param point The point being checked.
     * @return true when the provided point is below this node.
     */
    boolean isBelow(SGDVector point) {
        return point.get(dimension - 1) < coord;
    }

    /**
     * Traverse the (sub)tree rooted at this node to see if its descendants are closer to the provided point.
     *
     * @param point The target point.
     * @param queue The priority queue used to maintain the k nearest neighbours.
     * @param isInitializing A flag which indicates if this method is being called during query initialization.
     */
    void nearest(SGDVector point, DistanceRecordBoundedMinHeap queue, boolean isInitializing) {
        // stop the recursion during initialization, as soon at the queue has reached is target capacity bound.
        if (isInitializing && queue.isFull()) {
            return;
        }
        
        // Get the distance between this node and the target point.
        double dist = DistanceType.getDistance(this.record.vector, point, distanceType);
        queue.boundedOffer(this.record, dist);

        // Determine if we must traverse this node's subtrees by computing the perpendicular distance between the point
        // and the axis this node makes to separate the points in this region.
        // When distPerp is less than or equal to the largest distance currently in the queue, both sides of the tree
        // must be considered.
        double distPerp = Math.abs(coord - point.get(dimension - 1));

        if (Double.compare(distPerp, queue.peek().dist) <= 0) {
            // Check both sides
            if (above != null) {
                above.nearest(point, queue, isInitializing);
            }
            if (below != null) {
                below.nearest(point, queue, isInitializing);
            }
        } else {
            // Only need to check one subtree, determine which one.
            if (point.get(dimension - 1) < coord) {
                if (below != null) {
                    below.nearest (point, queue, isInitializing);
                }
            } else {
                if (above != null) {
                    above.nearest (point, queue, isInitializing);
                }
            }
        }
    }

}
