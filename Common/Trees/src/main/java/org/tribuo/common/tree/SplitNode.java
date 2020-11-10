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

import java.util.Objects;

/**
 * An immutable {@link Node} with a split and two child nodes.
 */
public class SplitNode<T extends Output<T>> implements Node<T> {
    private static final long serialVersionUID = 3L;

    private final Node<T> greaterThan;

    private final Node<T> lessThanOrEqual;

    private final int splitFeature;

    private final double splitValue;

    private final double impurity;

    /**
     * Constructs a split node with the specified split value, feature id, impurity and child nodes.
     * @param splitValue The feature value to split on.
     * @param featureID The feature id number.
     * @param impurity The impurity of this node at training time.
     * @param greaterThan The node to take if the feature value is greater than the split value.
     * @param lessThanOrEqual The node to take if the feature value is less than or equal to the split value.
     */
    public SplitNode(double splitValue, int featureID, double impurity, Node<T> greaterThan, Node<T> lessThanOrEqual) {
        this.splitValue = splitValue;
        this.splitFeature = featureID;
        this.impurity = impurity;
        this.greaterThan = greaterThan;
        this.lessThanOrEqual = lessThanOrEqual;
    }

    /**
     * Return the appropriate child node. If the splitFeature is not present in
     * the example it's value is treated as zero.
     *
     * @param e The example to inspect.
     * @return The corresponding child node.
     */
    @Override
    public Node<T> getNextNode(SparseVector e) {
        double feature = e.get(splitFeature);
        if (feature > splitValue) {
            return greaterThan;
        } else {
            return lessThanOrEqual;
        }
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public double getImpurity() {
        return impurity;
    }

    @Override
    public Node<T> copy() {
        return new SplitNode<>(splitValue,splitFeature,impurity,greaterThan.copy(),lessThanOrEqual.copy());
    }

    /**
     * Gets the feature ID that this node uses for splitting.
     * @return The feature ID.
     */
    public int getFeatureID() {
        return splitFeature;
    }

    /**
     * The threshold value.
     * @return The threshold value.
     */
    public double splitValue() {
        return splitValue;
    }

    /**
     * The node used if the value is greater than the splitValue.
     * @return The greater than node.
     */
    public Node<T> getGreaterThan() {
        return greaterThan;
    }

    /**
     * The node used if the value is less than or equal to the splitValue.
     * @return The less than or equal to node.
     */
    public Node<T> getLessThanOrEqual() {
        return lessThanOrEqual;
    }

    @Override
    public String toString() {
        return "SplitNode(feature="+splitFeature+",value="+splitValue+",impurity="+impurity+",\n\t\tleft="+lessThanOrEqual.toString()+",\n\t\tright="+greaterThan.toString()+")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SplitNode<?> splitNode = (SplitNode<?>) o;
        return splitFeature == splitNode.splitFeature &&
                Double.compare(splitNode.splitValue, splitValue) == 0 &&
                Double.compare(splitNode.impurity, impurity) == 0 &&
                greaterThan.equals(splitNode.greaterThan) &&
                lessThanOrEqual.equals(splitNode.lessThanOrEqual);
    }

    @Override
    public int hashCode() {
        return Objects.hash(greaterThan, lessThanOrEqual, splitFeature, splitValue, impurity);
    }
}

