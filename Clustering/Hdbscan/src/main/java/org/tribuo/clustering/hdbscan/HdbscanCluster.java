/*
 * Copyright (c) 2021, 2026, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Collections;
import java.util.List;
import java.util.TreeSet;

/**
 * An HDBSCAN* cluster, which encapsulates the attributes and functions of a cluster of points, throughout
 * the construction of the HDBSCAN* hierarchy.
 */
abstract sealed class HdbscanCluster implements Comparable<HdbscanCluster> {
    private HdbscanCluster() { }

    static final HdbscanCluster NOT_A_CLUSTER = new NotACluster();

    static HdbscanCluster create(int label, HdbscanCluster parent, double birthLevel, int numPoints) {
        return new RegularCluster(label, parent, birthLevel, numPoints);
    }

    /**
     * Detaches points and creates a child cluster, or assigns the points to noise.
     * @param points The points to remove from this cluster.
     * @param clusterLabels The point labels to update.
     * @param clusterLabel The new cluster label, or zero for noise.
     * @param edgeWeight The edge weight at which the points leave this cluster.
     * @return The new cluster, or {@link #NOT_A_CLUSTER} for noise.
     */
    abstract HdbscanCluster createNewCluster(TreeSet<Integer> points, int[] clusterLabels, int clusterLabel, double edgeWeight);

    /**
     * Removes points at the supplied edge level and updates this cluster's stability.
     * @param numPoints The number of points to remove.
     * @param level The edge weight at which the points leave this cluster.
     */
    abstract void detachPoints(int numPoints, double level);

    /**
     * Finalizes local split information, including for a root without children.
     */
    abstract void finishPropagation();

    /**
     * Propagates a finalized cluster into its real parent; must not be called on the root.
     */
    abstract void propagateToParent();

    /**
     * Records that this cluster has a child.
     */
    abstract void markHasChildren();

    /**
     * Accumulates the split level, stability and selected descendants propagated by a child.
     * @param lowestSplitLevel The lowest split level among the child's descendants.
     * @param stability The stability contributed by the child or its selected descendants.
     * @param descendants The selected clusters to append to this cluster's propagated descendants.
     */
    abstract void acceptPropagation(double lowestSplitLevel, double stability, List<HdbscanCluster> descendants);

    /**
     * Returns this cluster's label.
     * @return The cluster label, or -1 for the sentinel.
     */
    abstract int getLabel();

    /**
     * Returns the cluster which split to create this cluster.
     * @return The parent cluster, the sentinel for the root, or null for the sentinel.
     */
    abstract HdbscanCluster getParent();

    /**
     * Returns the lowest split level propagated from this cluster's descendants.
     * @return The lowest split level, or {@link Double#MAX_VALUE} if unset.
     */
    abstract double getPropagatedLowestChildSplitLevel();

    /**
     * Returns the selected clusters propagated from this cluster's children.
     * @return The propagated descendants.
     */
    abstract List<HdbscanCluster> getPropagatedDescendants();

    /**
     * Reports whether this cluster has children.
     * @return True if this cluster has children.
     */
    abstract boolean hasChildren();

    /**
     * Returns the hierarchy level at which this cluster was created.
     * @return The hierarchy level.
     */
    abstract int getHierarchyLevel();

    /**
     * Sets the hierarchy level at which this cluster was created.
     * @param hierarchyLevel The hierarchy level.
     */
    abstract void setHierarchyLevel(int hierarchyLevel);

    @Override
    public final int compareTo(HdbscanCluster that) {
        // The priority queue processes larger labels before their parents.
        return Integer.compare(that.getLabel(), getLabel());
    }

    private static final class RegularCluster extends HdbscanCluster {
        private final int label;

        private final double birthLevel;

        private double splitLevel;

        private int numPoints;

        private double stability;

        private double propagatedStability;

        private double propagatedLowestChildSplitLevel;

        private final HdbscanCluster parent;

        private boolean hasChildren;

        private final List<HdbscanCluster> propagatedDescendants;

        private int hierarchyLevel = 0;

        /**
         * Creates a new Cluster.
         * @param label The cluster label, which should be globally unique
         * @param parent The parent cluster, the cluster which split to create this cluster
         * @param birthLevel The EMST edge weight level at which this cluster first appeared
         * @param numPoints The initial number of points in this cluster
         */
        private RegularCluster(int label, HdbscanCluster parent, double birthLevel, int numPoints) {
            this.label = label;
            this.birthLevel = birthLevel;
            this.splitLevel = 0;
            this.numPoints = numPoints;
            this.stability = 0;
            this.propagatedStability = 0;
            this.propagatedLowestChildSplitLevel = Double.MAX_VALUE;

            this.parent = parent;
            if (this.parent != null && this.parent != NOT_A_CLUSTER) {
                this.parent.markHasChildren();
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
        @Override
        HdbscanCluster createNewCluster(TreeSet<Integer> points, int[] clusterLabels, int clusterLabel, double edgeWeight) {
            for (int point : points) {
                clusterLabels[point] = clusterLabel;
            }
            detachPoints(points.size(), edgeWeight);

            if (clusterLabel != HdbscanTrainer.OUTLIER_NOISE_CLUSTER_LABEL) {
                return new RegularCluster(clusterLabel, this, edgeWeight, points.size());
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
        @Override
        void detachPoints(int numPoints, double level) {
            this.numPoints-=numPoints;
            this.stability+=(numPoints * (1/level - 1/this.birthLevel));

            if (this.numPoints == 0) {
                this.splitLevel = level;
            } else if (this.numPoints < 0) {
                throw new IllegalStateException("Cluster cannot have less than 0 points.");
            }
        }

        @Override
        void finishPropagation() {
            if (propagatedLowestChildSplitLevel == Double.MAX_VALUE) {
                propagatedLowestChildSplitLevel = splitLevel;
            }
        }

        @Override
        void propagateToParent() {
            if (parent != null) {
                // Prefer this cluster over its descendants on a stability tie.
                if (!hasChildren || stability >= propagatedStability) {
                    parent.acceptPropagation(propagatedLowestChildSplitLevel, stability, Collections.singletonList(this));
                } else {
                    parent.acceptPropagation(propagatedLowestChildSplitLevel, propagatedStability, propagatedDescendants);
                }
            }
        }

        @Override
        void markHasChildren() {
            hasChildren = true;
        }

        @Override
        void acceptPropagation(double lowestSplitLevel, double stability, List<HdbscanCluster> descendants) {
            if (lowestSplitLevel < propagatedLowestChildSplitLevel) {
                propagatedLowestChildSplitLevel = lowestSplitLevel;
            }
            propagatedStability += stability;
            propagatedDescendants.addAll(descendants);
        }

        @Override
        int getLabel() {
            return this.label;
        }

        @Override
        HdbscanCluster getParent() {
            return this.parent;
        }

        @Override
        double getPropagatedLowestChildSplitLevel() {
            return this.propagatedLowestChildSplitLevel;
        }

        @Override
        List<HdbscanCluster> getPropagatedDescendants() {
            return this.propagatedDescendants;
        }

        @Override
        boolean hasChildren() {
            return this.hasChildren;
        }

        @Override
        int getHierarchyLevel() {
            return hierarchyLevel;
        }

        @Override
        void setHierarchyLevel(int hierarchyLevel) {
            this.hierarchyLevel = hierarchyLevel;
        }

    }

    /**
     * Stateless marker for noise and the root's absent parent.
     */
    private static final class NotACluster extends HdbscanCluster {
        @Override
        HdbscanCluster createNewCluster(TreeSet<Integer> points, int[] labels, int label, double level) {
            throw immutable();
        }

        @Override
        void detachPoints(int numPoints, double level) {
            throw immutable();
        }

        @Override
        void finishPropagation() {
            throw immutable();
        }

        @Override
        void propagateToParent() {
            throw immutable();
        }

        @Override
        void markHasChildren() {
            throw immutable();
        }

        @Override
        void acceptPropagation(double level, double stability, List<HdbscanCluster> descendants) {
            throw immutable();
        }

        @Override
        void setHierarchyLevel(int level) {
            throw immutable();
        }

        @Override
        int getLabel() {
            return -1;
        }

        @Override
        HdbscanCluster getParent() {
            return null;
        }

        @Override
        double getPropagatedLowestChildSplitLevel() {
            return Double.MAX_VALUE;
        }

        @Override
        List<HdbscanCluster> getPropagatedDescendants() {
            return Collections.emptyList();
        }

        @Override
        boolean hasChildren() {
            return false;
        }

        @Override
        int getHierarchyLevel() {
            return 0;
        }

        private UnsupportedOperationException immutable() {
            return new UnsupportedOperationException("NOT_A_CLUSTER cannot be modified");
        }
    }
}
