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

import java.util.HashMap;
import java.util.Map;

/**
 * A leaf {@link Node} that can create a prediction.
 */
public class LeafNode<T extends Output<T>> implements Node<T> {
    private static final long serialVersionUID = 4L;

    private final double impurity;

    private final T output;
    private final Map<String,T> scores;
    private final boolean generatesProbabilities;

    public LeafNode(double impurity, T output, Map<String,T> scores, boolean generatesProbabilities) {
        this.impurity = impurity;
        this.output = output;
        this.scores = scores;
        this.generatesProbabilities = generatesProbabilities;
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

    public T getOutput() {
        return output;
    }

    public Map<String,T> getDistribution() {
        return scores;
    }

    public Prediction<T> getPrediction(int numUsed, Example<T> example) {
        return new Prediction<>(output,scores,numUsed,example,generatesProbabilities);
    }

    @Override
    public String toString() {
        return "LeafNode(impurity="+impurity+",output="+output.toString()+",scores="+scores.toString()+",probability="+generatesProbabilities+")";
    }

}
