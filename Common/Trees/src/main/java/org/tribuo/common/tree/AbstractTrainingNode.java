/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.tree;

import org.tribuo.Output;
import org.tribuo.math.la.SparseVector;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Base class for decision tree nodes used at training time.
 */
public abstract class AbstractTrainingNode<T extends Output<T>> implements Node<T> {

    /**
     * Default buffer size used in the split operation.
     */
    protected static final int DEFAULT_SIZE = 16;

    protected final int depth;

    protected final int numExamples;

    protected final LeafDeterminer leafDeterminer;

    protected boolean split;

    protected int splitID;

    protected double splitValue;

    protected double impurityScore;
    
    protected Node<T> greaterThan;
    
    protected Node<T> lessThanOrEqual;

    /**
     * Builds an abstract training node.
     * @param depth The depth of this node.
     * @param numExamples The number of examples in this node.
     * @param leafDeterminer The parameters which determine if the node forms a leaf.
     */
    protected AbstractTrainingNode(int depth, int numExamples, LeafDeterminer leafDeterminer) {
        this.depth = depth;
        this.numExamples = numExamples;
        this.leafDeterminer = leafDeterminer;
    }

    /**
     * Builds next level of a tree.
     * @param featureIDs Indices of the features available in this split.
     * @param rng Splittable random number generator.
     * @param useRandomSplitPoints Whether to choose split points for features at random.
     * @return A possibly empty list of TrainingNodes.
     */
    public abstract List<AbstractTrainingNode<T>> buildTree(int[] featureIDs, SplittableRandom rng,
                                                            boolean useRandomSplitPoints);

    /**
     * Converts a tree from a training representation to the final inference time representation.
     * @return The converted subtree.
     */
    public abstract Node<T> convertTree();

    /**
     * The sum of the weights associated with this node's examples.
     * @return the sum of the weights associated with this node's examples.
     */
    public abstract float getWeightSum();

    /**
     * The depth of this node in the tree.
     * @return The depth.
     */
    public int getDepth() {
        return depth;
    }

    /**
     * Determines whether the node to be created should be a {@link LeafNode}.
     * @param impurityScore impurity score for the new node.
     * @param weightSum total example weight for the new node.
     * @return Whether the new node should be a {@link LeafNode}.
     */
    public boolean shouldMakeLeaf(double impurityScore, float weightSum) {
        return ((Math.abs(impurityScore) < 1e-15) ||
                (depth + 1 >= leafDeterminer.getMaxDepth()) ||
                (weightSum < leafDeterminer.getMinChildWeight()));
    }

    /**
     * Transforms an {@link AbstractTrainingNode} into a {@link SplitNode}
     * @return A {@link SplitNode}
     */
    public SplitNode<T> createSplitNode() {
        Node<T> newGreaterThan = greaterThan;
        Node<T> newLessThan = lessThanOrEqual;

        // split node
        if (greaterThan instanceof AbstractTrainingNode) {
            AbstractTrainingNode<T> abstractGreaterThan = (AbstractTrainingNode<T>) greaterThan;
            newGreaterThan = abstractGreaterThan.convertTree();
        }

        if (lessThanOrEqual instanceof AbstractTrainingNode) {
            AbstractTrainingNode<T> abstractLessThan = (AbstractTrainingNode<T>) lessThanOrEqual;
            newLessThan = abstractLessThan.convertTree();
        }
        return new SplitNode<>(splitValue,splitID,getImpurity(),newGreaterThan,newLessThan);
    }

    @Override
    public Node<T> getNextNode(SparseVector example) {
        if (split) {
            double feature = example.get(splitID);
            if (feature > splitValue) {
                return greaterThan;
            } else {
                return lessThanOrEqual;
            }
        } else {
            return null;
        }
    }

    /**
     * The number of training examples in this node.
     * @return The number of training examples in this node.
     */
    public int getNumExamples() {
        return numExamples;
    }

    @Override
    public boolean isLeaf() {
        return !split;
    }

    @Override
    public Node<T> copy() {
        throw new UnsupportedOperationException("Copy is not supported on training nodes.");
    }

    /**
     * Contains parameters needed to determine whether a node is a leaf.
     */
    // Will be a record one day.
    public static class LeafDeterminer {
        private final int maxDepth;
        private final float minChildWeight;
        private final float scaledMinImpurityDecrease;

        /**
         * Constructs a leaf determiner using the supplied parameters.
         * @param maxDepth The maximum tree depth.
         * @param minChildWeight The minimum example weight of each child node.
         * @param scaledMinImpurityDecrease  The scaled minimum impurity decrease necessary to split a node.
         */
        public LeafDeterminer(int maxDepth, float minChildWeight, float scaledMinImpurityDecrease) {
            this.maxDepth = maxDepth;
            this.minChildWeight = minChildWeight;
            this.scaledMinImpurityDecrease = scaledMinImpurityDecrease;
        }

        /**
         * Gets the maximum tree depth.
         * @return The maximum tree depth.
         */
        public int getMaxDepth() {
            return maxDepth;
        }

        /**
         * Gets the minimum example weight of a child node.
         * @return The mimimum weight of a child node.
         */
        public float getMinChildWeight() {
            return minChildWeight;
        }

        /**
         * Gets the minimum impurity decrease necessary to split a node.
         * @return The minimum impurity decrease to split a node.
         */
        public float getScaledMinImpurityDecrease() {
            return scaledMinImpurityDecrease;
        }
    }

}
