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

import com.google.protobuf.Any;
import org.tribuo.Output;
import org.tribuo.common.tree.protos.SplitNodeProto;
import org.tribuo.common.tree.protos.TreeNodeProto;
import org.tribuo.math.la.SparseVector;

import java.util.Objects;

/**
 * An immutable {@link Node} with a split and two child nodes.
 */
public class SplitNode<T extends Output<T>> implements Node<T> {
    private static final long serialVersionUID = 3L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

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

    TreeNodeProto serialize(int parentIdx, int curIdx, int greaterThanIdx, int lessThanOrEqualIdx) {
        SplitNodeProto.Builder nodeBuilder = SplitNodeProto.newBuilder();
        nodeBuilder.setParentIdx(parentIdx);
        nodeBuilder.setCurIdx(curIdx);
        nodeBuilder.setGreaterThanIdx(greaterThanIdx);
        nodeBuilder.setLessThanOrEqualIdx(lessThanOrEqualIdx);
        nodeBuilder.setSplitFeatureIdx(splitFeature);
        nodeBuilder.setSplitValue(splitValue);
        nodeBuilder.setImpurity(impurity);

        TreeNodeProto.Builder builder = TreeNodeProto.newBuilder();
        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(LeafNode.class.getName());
        builder.setSerializedData(Any.pack(nodeBuilder.build()));

        return builder.build();
    }

    static final class SplitNodeBuilder<T extends Output<T>> extends TreeModel.NodeBuilder implements Node<T> {

        private final int parentIdx;
        private final int curIdx;
        private final int greaterThanIdx;
        private final int lessThanOrEqualIdx;
        private final int splitFeature;
        private final double splitValue;
        private final double impurity;

        private Node<T> greaterThan;
        private Node<T> lessThanOrEqual;

        SplitNodeBuilder(SplitNodeProto proto) {
            this.parentIdx = proto.getParentIdx();
            this.curIdx = proto.getCurIdx();
            this.greaterThanIdx = proto.getGreaterThanIdx();
            this.lessThanOrEqualIdx = proto.getLessThanOrEqualIdx();
            this.splitFeature = proto.getSplitFeatureIdx();
            this.splitValue = proto.getSplitValue();
            this.impurity = proto.getImpurity();
        }

        SplitNodeBuilder(int parentIdx, int curIdx, int greaterThanIdx, int lessThanOrEqualIdx, int splitFeature, double splitValue, double impurity) {
            this.parentIdx = parentIdx;
            this.curIdx = curIdx;
            this.greaterThanIdx = greaterThanIdx;
            this.lessThanOrEqualIdx = lessThanOrEqualIdx;
            this.splitFeature = splitFeature;
            this.splitValue = splitValue;
            this.impurity = impurity;
        }

        @Override
        public boolean isLeaf() {
            return false;
        }

        @Override
        public Node<T> getNextNode(SparseVector example) {
            return null;
        }

        @Override
        public double getImpurity() {
            return impurity;
        }

        @Override
        public SplitNodeBuilder<T> copy() {
            return new SplitNodeBuilder<>(parentIdx, curIdx, greaterThanIdx, lessThanOrEqualIdx, splitFeature, splitValue, impurity);
        }

        boolean canBuild() {
            return greaterThan != null && lessThanOrEqual != null;
        }

        SplitNode<T> build() {
            if (!canBuild()) {
                throw new IllegalStateException("Not ready to build this split node, missing the children pointers");
            }
            return new SplitNode<>(splitValue,splitFeature,impurity,greaterThan,lessThanOrEqual);
        }

        void setGreaterThan(Node<T> greaterThan) {
            if (this.greaterThan == null) {
                this.greaterThan = greaterThan;
            } else {
                throw new IllegalStateException("Invalid protobuf, multiple nodes mapped to the greaterThanIdx");
            }
        }

        void setLessThanOrEqual(Node<T> lessThanOrEqual) {
            if (this.lessThanOrEqual == null) {
                this.lessThanOrEqual = lessThanOrEqual;
            } else {
                throw new IllegalStateException("Invalid protobuf, multiple nodes mapped to the lessThanOrEqualIdx");
            }
        }

        int getGreaterThanIdx() {
            return greaterThanIdx;
        }

        int getLessThanOrEqualIdx() {
            return lessThanOrEqualIdx;
        }

        int getParentIdx() {
            return parentIdx;
        }

        int getCurIdx() {
            return curIdx;
        }
    }
}

