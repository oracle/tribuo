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

package org.tribuo.regression.rtree;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.common.tree.LeafNode;
import org.tribuo.common.tree.Node;
import org.tribuo.common.tree.SplitNode;
import org.tribuo.common.tree.TreeModel;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.Regressor.DimensionTuple;

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
 * A {@link Model} wrapped around a list of decision tree root {@link Node}s used
 * to generate independent predictions for each dimension in a regression.
 */
public final class IndependentRegressionTreeModel extends TreeModel<Regressor> {
    private static final long serialVersionUID = 1L;

    private final Map<String,Node<Regressor>> roots;

    IndependentRegressionTreeModel(String name, ModelProvenance description,
                                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, boolean generatesProbabilities,
                                          Map<String,Node<Regressor>> roots) {
        super(name, description, featureIDMap, outputIDInfo, generatesProbabilities, gatherActiveFeatures(featureIDMap,roots));
        this.roots = roots;
    }

    private static Map<String,List<String>> gatherActiveFeatures(ImmutableFeatureMap fMap, Map<String,Node<Regressor>> roots) {
        HashMap<String,List<String>> outputMap = new HashMap<>();
        for (Map.Entry<String,Node<Regressor>> e : roots.entrySet()) {
            Set<String> activeFeatures = new LinkedHashSet<>();

            Queue<Node<Regressor>> nodeQueue = new LinkedList<>();

            nodeQueue.offer(e.getValue());

            while (!nodeQueue.isEmpty()) {
                Node<Regressor> node = nodeQueue.poll();
                if ((node != null) && (!node.isLeaf())) {
                    SplitNode<Regressor> splitNode = (SplitNode<Regressor>) node;
                    String featureName = fMap.get(splitNode.getFeatureID()).getName();
                    activeFeatures.add(featureName);
                    nodeQueue.offer(splitNode.getGreaterThan());
                    nodeQueue.offer(splitNode.getLessThanOrEqual());
                }
            }
            outputMap.put(e.getKey(), new ArrayList<>(activeFeatures));
        }
        return outputMap;
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        //
        // Ensures we handle collisions correctly
        SparseVector vec = SparseVector.createSparseVector(example,featureIDMap,false);
        if (vec.numActiveElements() == 0) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }

        List<Prediction<Regressor>> predictionList = new ArrayList<>();
        for (Map.Entry<String,Node<Regressor>> e : roots.entrySet()) {
            Node<Regressor> oldNode = e.getValue();
            Node<Regressor> curNode = e.getValue();

            while (curNode != null) {
                oldNode = curNode;
                curNode = oldNode.getNextNode(vec);
            }

            //
            // oldNode must be a LeafNode.
            predictionList.add(((LeafNode<Regressor>) oldNode).getPrediction(vec.numActiveElements(), example));
        }
        return combine(predictionList);
    }

    @Override
    public Map<String, List<Pair<String,Double>>> getTopFeatures(int n) {
        int maxFeatures = n < 0 ? featureIDMap.size() : n;

        Map<String, List<Pair<String, Double>>> map = new HashMap<>();
        Map<String, Integer> featureCounts = new HashMap<>();
        Queue<Node<Regressor>> nodeQueue = new LinkedList<>();

        for (Map.Entry<String,Node<Regressor>> e : roots.entrySet()) {
            featureCounts.clear();
            nodeQueue.clear();

            nodeQueue.offer(e.getValue());

            while (!nodeQueue.isEmpty()) {
                Node<Regressor> node = nodeQueue.poll();
                if ((node != null) && !node.isLeaf()) {
                    SplitNode<Regressor> splitNode = (SplitNode<Regressor>) node;
                    String featureName = featureIDMap.get(splitNode.getFeatureID()).getName();
                    featureCounts.put(featureName, featureCounts.getOrDefault(featureName, 0) + 1);
                    nodeQueue.offer(splitNode.getGreaterThan());
                    nodeQueue.offer(splitNode.getLessThanOrEqual());
                }
            }

            Comparator<Pair<String, Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));
            PriorityQueue<Pair<String, Double>> q = new PriorityQueue<>(maxFeatures, comparator);

            for (Map.Entry<String, Integer> featureCount : featureCounts.entrySet()) {
                Pair<String, Double> cur = new Pair<>(featureCount.getKey(), (double) featureCount.getValue());
                if (q.size() < maxFeatures) {
                    q.offer(cur);
                } else if (comparator.compare(cur, q.peek()) > 0) {
                    q.poll();
                    q.offer(cur);
                }
            }
            List<Pair<String, Double>> list = new ArrayList<>();
            while (q.size() > 0) {
                list.add(q.poll());
            }
            Collections.reverse(list);

            map.put(e.getKey(), list);
        }

        return map;
    }

    @Override
    public Optional<Excuse<Regressor>> getExcuse(Example<Regressor> example) {
        SparseVector vec = SparseVector.createSparseVector(example, featureIDMap, false);
        if (vec.numActiveElements() == 0) {
            return Optional.empty();
        }

        List<String> list = new ArrayList<>();
        List<Prediction<Regressor>> predList = new ArrayList<>();
        Map<String, List<Pair<String, Double>>> map = new HashMap<>();

        for (Map.Entry<String,Node<Regressor>> e : roots.entrySet()) {
            list.clear();

            //
            // Ensures we handle collisions correctly
            Node<Regressor> oldNode = e.getValue();
            Node<Regressor> curNode = e.getValue();

            while (curNode != null) {
                oldNode = curNode;
                if (oldNode instanceof SplitNode) {
                    SplitNode<?> node = (SplitNode<?>) curNode;
                    list.add(featureIDMap.get(node.getFeatureID()).getName());
                }
                curNode = oldNode.getNextNode(vec);
            }

            //
            // oldNode must be a LeafNode.
            predList.add(((LeafNode<Regressor>) oldNode).getPrediction(vec.numActiveElements(), example));

            List<Pair<String, Double>> pairs = new ArrayList<>();
            int i = list.size() + 1;
            for (String s : list) {
                pairs.add(new Pair<>(s, i + 0.0));
                i--;
            }

            map.put(e.getKey(), pairs);
        }
        Prediction<Regressor> combinedPrediction = combine(predList);

        return Optional.of(new Excuse<>(example,combinedPrediction,map));
    }

    @Override
    protected IndependentRegressionTreeModel copy(String newName, ModelProvenance newProvenance) {
        Map<String,Node<Regressor>> newRoots = new HashMap<>();
        for (Map.Entry<String,Node<Regressor>> e : roots.entrySet()) {
            newRoots.put(e.getKey(),e.getValue().copy());
        }
        return new IndependentRegressionTreeModel(newName,newProvenance,featureIDMap,outputIDInfo,generatesProbabilities,newRoots);
    }

    private Prediction<Regressor> combine(List<Prediction<Regressor>> predictions) {
        DimensionTuple[] tuples = new DimensionTuple[predictions.size()];
        int numUsed = 0;
        int i = 0;
        for (Prediction<Regressor> p : predictions) {
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            Regressor output = p.getOutput();
            if (output instanceof DimensionTuple) {
                tuples[i] = (DimensionTuple)output;
            } else {
                throw new IllegalStateException("All the leaves should contain DimensionTuple not Regressor");
            }
            i++;
        }

        Example<Regressor> example = predictions.get(0).getExample();
        return new Prediction<>(new Regressor(tuples),numUsed,example);
    }

    @Override
    public Set<String> getFeatures() {
        Set<String> features = new HashSet<>();

        Queue<Node<Regressor>> nodeQueue = new LinkedList<>();

        for (Map.Entry<String,Node<Regressor>> e : roots.entrySet()) {
            nodeQueue.offer(e.getValue());

            while (!nodeQueue.isEmpty()) {
                Node<Regressor> node = nodeQueue.poll();
                if ((node != null) && !node.isLeaf()) {
                    SplitNode<Regressor> splitNode = (SplitNode<Regressor>) node;
                    features.add(featureIDMap.get(splitNode.getFeatureID()).getName());
                    nodeQueue.offer(splitNode.getGreaterThan());
                    nodeQueue.offer(splitNode.getLessThanOrEqual());
                }
            }
        }

        return features;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String,Node<Regressor>> curRoot : roots.entrySet()) {
            sb.append("Output '");
            sb.append(curRoot.getKey());
            sb.append("' - tree = ");
            sb.append(curRoot.getValue().toString());
            sb.append('\n');
        }
        return "IndependentTreeModel(description="+provenance.toString()+",\n"+sb.toString()+")";
    }

}
