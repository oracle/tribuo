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

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

/**
 * An HDBSCAN* cluster, which encapsulates the attributes and functions of a cluster of points, throughout
 * the construction of the HDBSCAN* hierarchy.
 */
final class HdbscanCluster implements Comparable<HdbscanCluster> {
    private final int label;

    private final double birthLevel;

    private double splitLevel;

    private int numPoints;

    private double stability;

    private double propagatedStability;

    private double propagatedLowestChildSplitLevel;

    private final HdbscanCluster parent;

    private boolean hasChildren;

    final private List<HdbscanCluster> propagatedDescendants;

    private int hierarchyLevel = 0;

    static final HdbscanCluster NOT_A_CLUSTER = new HdbscanCluster(-1, null, Double.NaN, 0);

    /**
     * Creates a new Cluster.
     * @param label The cluster label, which should be globally unique
     * @param parent The parent cluster, the cluster which split to create this cluster
     * @param birthLevel The EMST edge weight level at which this cluster first appeared
     * @param numPoints The initial number of points in this cluster
     */
    HdbscanCluster(int label, HdbscanCluster parent, double birthLevel, int numPoints) {
        this.label = label;
        this.birthLevel = birthLevel;
        this.splitLevel = 0;
        this.numPoints = numPoints;
        this.stability = 0;
        this.propagatedStability = 0;
        this.propagatedLowestChildSplitLevel = Double.MAX_VALUE;

        this.parent = parent;
        if (this.parent != null) {
            this.parent.hasChildren = true;
        }
        this.hasChildren = false;
        this.propagatedDescendants = new ArrayList<>(1);
    }

    /**
     * Creates a new Cluster, provided the clusterId is not 0 (noise), and removes the set of points from
     * the parent Cluster.
     * @param points The set of points to be in the new Cluster
     * @param clusterLabels An array of cluster labels, which will be modified
     * @param clusterLabel The label of the new Cluster
     * @param edgeWeight The edge weight at which to remove the points from their previous Cluster
     * @return The new Cluster, or the {@link #NOT_A_CLUSTER} if the clusterId was 0 (Noise)
     */
    HdbscanCluster createNewCluster(TreeSet<Integer> points, int[] clusterLabels, int clusterLabel, double edgeWeight) {
        for (int point : points) {
            clusterLabels[point] = clusterLabel;
        }
        detachPoints(points.size(), edgeWeight);

        if (clusterLabel != HdbscanTrainer.OUTLIER_NOISE_CLUSTER_LABEL) {
            return new HdbscanCluster(clusterLabel, this, edgeWeight, points.size());
        } else {
            return NOT_A_CLUSTER;
        }
    }

    /**
     * Removes the specified number of points from this cluster at the given edge level, which will
     * update the stability of this cluster and potentially cause cluster death.
     * @param numPoints The number of points to remove from the cluster
     * @param level The EMST edge weight level at which to remove these points
     */
    void detachPoints(int numPoints, double level) {
        this.numPoints-=numPoints;
        this.stability+=(numPoints * (1/level - 1/this.birthLevel));

        if (this.numPoints == 0) {
            this.splitLevel = level;
        } else if (this.numPoints < 0) {
            throw new IllegalStateException("Cluster cannot have less than 0 points.");
        }
    }

    /**
     * This cluster will propagate itself to its parent, if it has no children. Otherwise, this cluster
     * propagates itself or its descendants preferring the former when its stability is greater or equal.
     * Additionally, this cluster propagates the lowest death level out of all its descendants to its parent.
     */
    void propagate() {
        if (this.parent != null) {
            // Propagate the lowest death level of any descendants:
            if (this.propagatedLowestChildSplitLevel == Double.MAX_VALUE) {
                this.propagatedLowestChildSplitLevel = this.splitLevel;
            }
            if (this.propagatedLowestChildSplitLevel < this.parent.propagatedLowestChildSplitLevel) {
                this.parent.propagatedLowestChildSplitLevel = this.propagatedLowestChildSplitLevel;
            }

            // If this cluster has no children, it must propagate itself:
            if (!this.hasChildren) {
                this.parent.propagatedStability+= this.stability;
                this.parent.propagatedDescendants.add(this);
            } else {
                // Chose the parent over descendants if there is a tie in stability:
                if (this.stability >= this.propagatedStability) {
                    this.parent.propagatedStability+= this.stability;
                    this.parent.propagatedDescendants.add(this);
                } else {
                    this.parent.propagatedStability+= this.propagatedStability;
                    this.parent.propagatedDescendants.addAll(this.propagatedDescendants);
                }
            }
        }
    }

    int getLabel() {
        return this.label;
    }

    HdbscanCluster getParent() {
        return this.parent;
    }

    double getPropagatedLowestChildSplitLevel() {
        return this.propagatedLowestChildSplitLevel;
    }

    List<HdbscanCluster> getPropagatedDescendants() {
        return this.propagatedDescendants;
    }

    boolean hasChildren() {
        return this.hasChildren;
    }

    int getHierarchyLevel() {
        return hierarchyLevel;
    }

    void setHierarchyLevel(int hierarchyLevel) {
        this.hierarchyLevel = hierarchyLevel;
    }

    @Override
    public int compareTo(HdbscanCluster that) {
        // This class is used in a PriorityQueue. When polling, we need the reverse of the natural ordering.
        int val = Integer.compare(this.label, that.label);
        if (val == 0) {
            return val;
        } else {
            return -val;
        }
    }
}