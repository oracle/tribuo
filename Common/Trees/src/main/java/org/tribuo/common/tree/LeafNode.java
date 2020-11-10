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

import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.math.la.SparseVector;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * An immutable leaf {@link Node} that can create a prediction.
 * <p>
 * {@link LeafNode#equals} uses the {@link Output#fullEquals(Output)} method
 * to determine equality of two leaves.
 */
public class LeafNode<T extends Output<T>> implements Node<T> {
    private static final long serialVersionUID = 4L;

    private final double impurity;

    private final T output;
    private final Map<String,T> scores;
    private final boolean generatesProbabilities;

    /**
     * Constructs a leaf node.
     * @param impurity The impurity value calculated at training time.
     * @param output The output value from this node.
     * @param scores The score map for the other outputs.
     * @param generatesProbabilities If the scores are probabilities.
     */
    public LeafNode(double impurity, T output, Map<String,T> scores, boolean generatesProbabilities) {
        this.impurity = impurity;
        this.output = output;
        this.scores = Collections.unmodifiableMap(scores);
        this.generatesProbabilities = generatesProbabilities;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LeafNode<?> leafNode = (LeafNode<?>) o;
        if (output.getClass().equals(leafNode.output.getClass())) {
            @SuppressWarnings("unchecked") //guarded by class check
            LeafNode<T> typedLeafNode = (LeafNode<T>) leafNode;
            // If the scores have the same keys.
            if (scores.keySet().equals(typedLeafNode.scores.keySet())) {
                // Check the values are the same.
                boolean valueEquals = true;
                for (Map.Entry<String,T> e : scores.entrySet()) {
                    valueEquals &= e.getValue().fullEquals(typedLeafNode.scores.get(e.getKey()));
                }
                // Check the rest of the object.
                return valueEquals &&
                        Double.compare(typedLeafNode.impurity, impurity) == 0 &&
                        generatesProbabilities == typedLeafNode.generatesProbabilities &&
                        output.fullEquals(typedLeafNode.output);
            }
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(impurity, output, scores, generatesProbabilities);
    }

    @Override
    public Node<T> getNextNode(SparseVector e) {
        return null;
    }
    
    @Override
    public boolean isLeaf() {
        return true;
    }

    @Override
    public double getImpurity() {
        return impurity;
    }

    @Override
    public LeafNode<T> copy() {
        return new LeafNode<>(impurity,output.copy(),new HashMap<>(scores),generatesProbabilities);
    }

    /**
     * Gets the output in this node.
     * @return The output.
     */
    public T getOutput() {
        return output;
    }

    /**
     * Gets the distribution over scores in this node.
     * @return The score distribution.
     */
    public Map<String,T> getDistribution() {
        return scores;
    }

    /**
     * Constructs a new prediction object based on this node's scores.
     * @param numUsed The number of features used.
     * @param example The example to be scored.
     * @return The prediction for the supplied example.
     */
    public Prediction<T> getPrediction(int numUsed, Example<T> example) {
        return new Prediction<>(output,scores,numUsed,example,generatesProbabilities);
    }

    @Override
    public String toString() {
        return "LeafNode(impurity="+impurity+",output="+output.toString()+",scores="+scores.toString()+",probability="+generatesProbabilities+")";
    }

}
