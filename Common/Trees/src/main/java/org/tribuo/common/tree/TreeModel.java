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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.SparseModel;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;

/**
 * A {@link Model} wrapped around a decision tree root {@link Node}.
 */
public class TreeModel<T extends Output<T>> extends SparseModel<T> {
    private static final long serialVersionUID = 3L;

    private final Node<T> root;

    /**
     * Constructs a trained decision tree model.
     * @param name The model name.
     * @param description The model provenance.
     * @param featureIDMap The feature id map.
     * @param outputIDInfo The output info.
     * @param generatesProbabilities Does this model emit probabilities.
     * @param root The root node of the tree.
     */
    TreeModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, boolean generatesProbabilities, Node<T> root) {
        super(name, description, featureIDMap, outputIDInfo, generatesProbabilities, gatherActiveFeatures(featureIDMap,root));
        this.root = root;
    }

    /**
     * Constructs a trained decision tree model.
     * <p>
     * Only used when the tree has multiple roots, should only be called from
     * subclasses when *all* other methods are overridden.
     * @param name The model name.
     * @param description The model provenance.
     * @param featureIDMap The feature id map.
     * @param outputIDInfo The output info.
     * @param generatesProbabilities Does this model emit probabilities.
     * @param activeFeatures The active feature set of the model.
     */
    protected TreeModel(String name, ModelProvenance description,
                        ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                        boolean generatesProbabilities, Map<String,List<String>> activeFeatures) {
        super(name, description, featureIDMap, outputIDInfo, generatesProbabilities, activeFeatures);
        this.root = null;
    }

    private static <T extends Output<T>> Map<String,List<String>> gatherActiveFeatures(ImmutableFeatureMap fMap, Node<T> root) {
        Set<String> activeFeatures = new LinkedHashSet<>();

        Queue<Node<T>> nodeQueue = new LinkedList<>();

        nodeQueue.offer(root);

        while (!nodeQueue.isEmpty()) {
            Node<T> node = nodeQueue.poll();
            if ((node != null) && (!node.isLeaf())) {
                SplitNode<T> splitNode = (SplitNode<T>) node;
                String featureName = fMap.get(splitNode.getFeatureID()).getName();
                activeFeatures.add(featureName);
                nodeQueue.offer(splitNode.getGreaterThan());
                nodeQueue.offer(splitNode.getLessThanOrEqual());
            }
        }
        return Collections.singletonMap(Model.ALL_OUTPUTS,new ArrayList<>(activeFeatures));
    }

    /**
     * Probes the tree to find the depth.
     * @return The depth of the tree.
     */
    public int getDepth() {
        return computeDepth(0,root);
    }

    protected static <T extends Output<T>> int computeDepth(int initialDepth, Node<T> root) {
        int maxDepth = initialDepth;
        Queue<Pair<Integer,Node<T>>> nodeQueue = new LinkedList<>();

        nodeQueue.offer(new Pair<>(initialDepth,root));

        while (!nodeQueue.isEmpty()) {
            Pair<Integer,Node<T>> nodePair = nodeQueue.poll();
            int curDepth = nodePair.getA() + 1;
            Node<T> node = nodePair.getB();
            if ((node != null) && !node.isLeaf()) {
                SplitNode<T> splitNode = (SplitNode<T>) node;
                Node<T> greaterThan = splitNode.getGreaterThan();
                Node<T> lessThan = splitNode.getLessThanOrEqual();
                if (greaterThan instanceof LeafNode) {
                    if (maxDepth < curDepth) {
                        maxDepth = curDepth;
                    }
                } else {
                    nodeQueue.offer(new Pair<>(curDepth,greaterThan));
                }
                if (lessThan instanceof LeafNode) {
                    if (maxDepth < curDepth) {
                        maxDepth = curDepth;
                    }
                } else {
                    nodeQueue.offer(new Pair<>(curDepth,lessThan));
                }
            }
        }

        return maxDepth;
    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        //
        // Ensures we handle collisions correctly
        SparseVector vec = SparseVector.createSparseVector(example,featureIDMap,false);
        if (vec.numActiveElements() == 0) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        Node<T> oldNode = root;
        Node<T> curNode = root;

        while (curNode != null) {
            oldNode = curNode;
            curNode = oldNode.getNextNode(vec);
        }

        //
        // oldNode must be a LeafNode.
        return ((LeafNode<T>) oldNode).getPrediction(vec.numActiveElements(),example);
    }

    @Override
    public Map<String, List<Pair<String,Double>>> getTopFeatures(int n) {
        int maxFeatures = n < 0 ? featureIDMap.size() : n;
        Map<String,Integer> featureCounts = new HashMap<>();

        Queue<Node<T>> nodeQueue = new LinkedList<>();

        nodeQueue.offer(root);

        while (!nodeQueue.isEmpty()) {
            Node<T> node = nodeQueue.poll();
            if ((node != null) && !node.isLeaf()) {
                SplitNode<T> splitNode = (SplitNode<T>) node;
                String featureName = featureIDMap.get(splitNode.getFeatureID()).getName();
                featureCounts.put(featureName, featureCounts.getOrDefault(featureName, 0) + 1);
                nodeQueue.offer(splitNode.getGreaterThan());
                nodeQueue.offer(splitNode.getLessThanOrEqual());
            }
        }

        Comparator<Pair<String,Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));
        PriorityQueue<Pair<String,Double>> q = new PriorityQueue<>(maxFeatures, comparator);

        for (Map.Entry<String, Integer> e : featureCounts.entrySet()) {
            Pair<String,Double> cur = new Pair<>(e.getKey(), (double) e.getValue());
            if (q.size() < maxFeatures) {
                q.offer(cur);
            } else if (comparator.compare(cur, q.peek()) > 0) {
                q.poll();
                q.offer(cur);
            }
        }
        List<Pair<String,Double>> list = new ArrayList<>();
        while (q.size() > 0) {
            list.add(q.poll());
        }
        Collections.reverse(list);

        Map<String,List<Pair<String,Double>>> map = new HashMap<>();
        map.put(Model.ALL_OUTPUTS, list);

        return map;
    }

    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        List<String> list = new ArrayList<>();
        //
        // Ensures we handle collisions correctly
        SparseVector vec = SparseVector.createSparseVector(example,featureIDMap,false);
        Node<T> oldNode = root;
        Node<T> curNode = root;

        while (curNode != null) {
            oldNode = curNode;
            if (oldNode instanceof SplitNode) {
                SplitNode<T> node = (SplitNode<T>) curNode;
                list.add(featureIDMap.get(node.getFeatureID()).getName());
            }
            curNode = oldNode.getNextNode(vec);
        }

        //
        // oldNode must be a LeafNode.
        Prediction<T> pred = ((LeafNode<T>) oldNode).getPrediction(vec.numActiveElements(),example);

        List<Pair<String,Double>> pairs = new ArrayList<>();
        int i = list.size() + 1;
        for (String s : list) {
            pairs.add(new Pair<>(s,i+0.0));
            i--;
        }

        Map<String,List<Pair<String,Double>>> map = new HashMap<>();
        map.put(Model.ALL_OUTPUTS,pairs);

        return Optional.of(new Excuse<>(example,pred,map));
    }

    @Override
    protected TreeModel<T> copy(String newName, ModelProvenance newProvenance) {
        return new TreeModel<>(newName,newProvenance,featureIDMap,outputIDInfo,generatesProbabilities,root.copy());
    }

    /**
     * Returns the set of features which are split on in this tree.
     * @return The feature names used by this tree.
     */
    public Set<String> getFeatures() {
        Set<String> features = new HashSet<>();

        Queue<Node<T>> nodeQueue = new LinkedList<>();

        nodeQueue.offer(root);

        while (!nodeQueue.isEmpty()) {
            Node<T> node = nodeQueue.poll();
            if ((node != null) && !node.isLeaf()) {
                SplitNode<T> splitNode = (SplitNode<T>) node;
                features.add(featureIDMap.get(splitNode.getFeatureID()).getName());
                nodeQueue.offer(splitNode.getGreaterThan());
                nodeQueue.offer(splitNode.getLessThanOrEqual());
            }
        }

        return features;
    }

    @Override
    public String toString() {
        return "TreeModel(description="+provenance.toString()+",\n\t\ttree="+root.toString()+")";
    }

    /**
     * Returns the root node of this tree.
     * @return The root node.
     */
    public Node<T> getRoot() {
        return root;
    }

}
