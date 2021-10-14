package org.tribuo.clustering.hdbscan;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

/**
 * An HDBSCAN* cluster, which encapsulates the attributes and functions of a cluster of points, throughout
 * the construction of the HDBSCAN* hierarchy.
 */
class HdbscanCluster {
    private final int label;

    private final double birthLevel;

    private double deathLevel;

    private int numPoints;

    private double stability;

    private double propagatedStability;

    private double propagatedLowestChildDeathLevel;

    private final HdbscanCluster parent;

    private boolean hasChildren;

    public List<HdbscanCluster> propagatedDescendants;

    int hierarchyLevel = 0;

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
        this.deathLevel = 0;
        this.numPoints = numPoints;
        this.stability = 0;
        this.propagatedStability = 0;
        this.propagatedLowestChildDeathLevel = Double.MAX_VALUE;

        this.parent = parent;
        if (this.parent != null)
            this.parent.hasChildren = true;
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
     * @return The new Cluster, or null if the clusterId was 0 (Noise)
     */
    HdbscanCluster createNewCluster(TreeSet<Integer> points, int[] clusterLabels, int clusterLabel, double edgeWeight) {
        for (int point : points) {
            clusterLabels[point] = clusterLabel;
        }
        detachPoints(points.size(), edgeWeight);

        if (clusterLabel != HdbscanTrainer.OUTLIER_NOISE_CLUSTER_LABEL)
            return new HdbscanCluster(clusterLabel, this, edgeWeight, points.size());
        else {
            return null;
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

        if (this.numPoints == 0)
            this.deathLevel = level;
        else if (this.numPoints < 0)
            throw new IllegalStateException("Cluster cannot have less than 0 points.");
    }

    /**
     * This cluster will propagate itself to its parent, if it has no children. Otherwise, this cluster
     * propagates itself or its descendants preferring the former when its stability is greater or equal.
     * Additionally, this cluster propagates the lowest death level out of all its descendants to its parent.
     */
    void propagate() {
        if (this.parent != null) {
            // Propagate the lowest death level of any descendants:
            if (this.propagatedLowestChildDeathLevel == Double.MAX_VALUE)
                this.propagatedLowestChildDeathLevel = this.deathLevel;
            if (this.propagatedLowestChildDeathLevel < this.parent.propagatedLowestChildDeathLevel)
                this.parent.propagatedLowestChildDeathLevel = this.propagatedLowestChildDeathLevel;

            // If this cluster has no children, it must propagate itself:
            if (!this.hasChildren) {
                this.parent.propagatedStability+= this.stability;
                this.parent.propagatedDescendants.add(this);
            }
            else {
                // Chose the parent over descendants if there is a tie in stability:
                if (this.stability >= this.propagatedStability) {
                    this.parent.propagatedStability+= this.stability;
                    this.parent.propagatedDescendants.add(this);
                }
                else {
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

    double getPropagatedLowestChildDeathLevel() {
        return this.propagatedLowestChildDeathLevel;
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
}