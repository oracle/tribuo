/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.SparseModel;
import org.tribuo.common.tree.protos.LeafNodeProto;
import org.tribuo.common.tree.protos.SplitNodeProto;
import org.tribuo.common.tree.protos.TreeModelProto;
import org.tribuo.common.tree.protos.TreeNodeProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.la.SparseVector;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
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

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

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

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     */
    @SuppressWarnings({"unchecked","rawtypes"}) // guarded by getClass to ensure all the output types are the same.
    public static TreeModel<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        TreeModelProto proto = message.unpack(TreeModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        Class<?> outputClass = carrier.outputDomain().getOutput(0).getClass();

        if (proto.getNodesCount() == 0) {
            throw new IllegalStateException("Invalid protobuf, tree must contain nodes");
        }

        List<TreeNodeProto> nodeProtos = proto.getNodesList();
        List<Node<?>> nodes = deserializeFromProtos(nodeProtos, (Class) outputClass);

        return new TreeModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),carrier.outputDomain(),carrier.generatesProbabilities(),nodes.get(0));
    }

    private static Node<?> deserializeNodeProto(TreeNodeProto proto) throws InvalidProtocolBufferException {
        int version = proto.getVersion();
        String className = proto.getClassName();
        Any message = proto.getSerializedData();
        if (message.is(SplitNodeProto.class)) {
            SplitNodeProto splitProto = message.unpack(SplitNodeProto.class);
            return new SplitNode.SplitNodeBuilder<>(splitProto);
        } else if (message.is(LeafNodeProto.class)) {
            LeafNodeProto leafProto = message.unpack(LeafNodeProto.class);
            return new LeafNode.LeafNodeBuilder<>(leafProto);
        } else {
            throw new IllegalStateException("Invalid protobuf, expected leaf or split node, found " + message.getTypeUrl());
        }
    }

    /**
     * Deserializes a list of node protos into a list of nodes, with the root in the first element.
     * @param nodeProtos The node protos to deserialize.
     * @return The nodes.
     * @throws InvalidProtocolBufferException If an unexpected proto is found.
     */
    //@SuppressWarnings({"unchecked","rawtypes"}) // guarded by getClass to ensure all the output types are the same.
    protected static <U extends Output<U>> List<Node<U>> deserializeFromProtos(List<TreeNodeProto> nodeProtos, Class<U> outputClass) throws InvalidProtocolBufferException {
        List<Node<U>> nodes = new ArrayList<>(nodeProtos.size());

        for (TreeNodeProto p : nodeProtos) {
            @SuppressWarnings("unchecked") // we'll catch this later with the getClass check
            Node<U> curNode = (Node<U>) deserializeNodeProto(p);
            nodes.add(curNode);
        }

        Queue<Node<U>> nodeQueue = new ArrayDeque<>();
        for (Node<U> node : nodes) {
            if (node instanceof LeafNode.LeafNodeBuilder) {
                nodeQueue.offer(node);
            }
        }

        while (!nodeQueue.isEmpty()) {
            Node<U> nodeBuilder = nodeQueue.poll();
            int curIdx = -1;
            Node<U> parent = null;
            Node<U> builtNode = null;
            if (nodeBuilder instanceof LeafNode.LeafNodeBuilder) {
                // build leaf node
                LeafNode.LeafNodeBuilder<U> builder = (LeafNode.LeafNodeBuilder<U>) nodeBuilder;
                LeafNode<U> leaf = builder.build();
                nodes.set(builder.getCurIdx(), leaf);
                builtNode = leaf;
                curIdx = builder.getCurIdx();
                // update parent
                int parentIdx = builder.getParentIdx();
                if (parentIdx != -1) {
                    parent = nodes.get(parentIdx);
                }
            } else if (nodeBuilder instanceof SplitNode.SplitNodeBuilder) {
                // build split node now the childred are ready
                SplitNode.SplitNodeBuilder<U> builder = (SplitNode.SplitNodeBuilder<U>) nodeBuilder;
                SplitNode<U> split = builder.build();
                nodes.set(builder.getCurIdx(), split);
                builtNode = split;
                curIdx = builder.getCurIdx();
                // update parent
                int parentIdx = builder.getParentIdx();
                if (parentIdx != -1) {
                    parent = nodes.get(parentIdx);
                }
            } else {
                throw new IllegalStateException("Invalid protobuf, found a constructed node was added to the build queue, found " + nodeBuilder.getClass());
            }
            if (parent instanceof SplitNode.SplitNodeBuilder) {
                SplitNode.SplitNodeBuilder<U> splitBuilder = (SplitNode.SplitNodeBuilder<U>) parent;
                if (curIdx == splitBuilder.getGreaterThanIdx()) {
                    splitBuilder.setGreaterThan(builtNode);
                } else if (curIdx == splitBuilder.getLessThanOrEqualIdx()) {
                    splitBuilder.setLessThanOrEqualIdx(builtNode);
                } else {
                    throw new IllegalStateException("Invalid protobuf, found a child node which didn't map into a parent");
                }
                // If we can build this split node pop it on the queue.
                if (splitBuilder.canBuild()) {
                    nodeQueue.offer(splitBuilder);
                }
            } else if (parent != null) {
                throw new IllegalStateException("Invalid protobuf, found a " + parent.getClass() + " when a SplitNodeBuilder was expected");
            }
        }

        for (Node<U> node : nodes) {
            if (!(node instanceof SplitNode || node instanceof LeafNode)) {
                throw new IllegalStateException("Invalid protobuf, found unbuilt node, " + node);
            } else if (node instanceof LeafNode) {
                U cur = ((LeafNode<U>) node).getOutput();
                if (!outputClass.isAssignableFrom(cur.getClass())) {
                    throw new IllegalStateException("Invalid protobuf, node output did not match output domain, found " + cur.getClass() + ", expected " + outputClass);
                }
            }
        }

        return nodes;
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

    /**
     * Counts the number of nodes in the tree rooted at the supplied node, including that node.
     * @param root The tree root.
     * @return The number of nodes.
     */
    public int countNodes(Node<T> root) {
        Queue<Node<T>> nodeQueue = new LinkedList<>();

        int counter = 0;
        nodeQueue.offer(root);

        while (!nodeQueue.isEmpty()) {
            Node<T> node = nodeQueue.poll();
            if (node != null) {
                counter++;
                if (!node.isLeaf()) {
                    SplitNode<T> splitNode = (SplitNode<T>) node;
                    nodeQueue.offer(splitNode.getGreaterThan());
                    nodeQueue.offer(splitNode.getLessThanOrEqual());
                }
            }
        }

        return counter;
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

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<T> carrier = createDataCarrier();

        TreeModelProto.Builder modelBuilder = TreeModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.addAllNodes(serializeToNodes(root));

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(TreeModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    protected List<TreeNodeProto> serializeToNodes(Node<T> root) {
        int numNodes = countNodes(root);
        TreeNodeProto[] protos = new TreeNodeProto[numNodes];

        int counter = 0;
        Queue<SerializationState<T>> nodeQueue = new ArrayDeque<>();
        nodeQueue.offer(new SerializationState<>(-1,counter,root));
        while (!nodeQueue.isEmpty()) {
            SerializationState<T> state = nodeQueue.poll();
            if (state.node instanceof SplitNode) {
                SplitNode<T> node = (SplitNode<T>) state.node;
                int greaterIdx = ++counter;
                int lessIdx = ++counter;
                TreeNodeProto proto = node.serialize(state.parentIdx, state.curIdx, greaterIdx, lessIdx);
                protos[state.curIdx] = proto;
                nodeQueue.offer(new SerializationState<>(state.curIdx, greaterIdx, node.getGreaterThan()));
                nodeQueue.offer(new SerializationState<>(state.curIdx, lessIdx, node.getLessThanOrEqual()));
            } else if (state.node instanceof LeafNode) {
                LeafNode<T> node = (LeafNode<T>) state.node;
                TreeNodeProto proto = node.serialize(state.parentIdx, state.curIdx);
                protos[state.curIdx] = proto;
            } else {
                throw new IllegalStateException("Invalid tree structure, contained a node which wasn't a SplitNode or a LeafNode, found " + state.node.getClass());
            }
        }

        return Arrays.asList(protos);
    }

    private static final class SerializationState<T extends Output<T>> {
        final int parentIdx;
        final int curIdx;
        final Node<T> node;

        SerializationState(int parentIdx, int curIdx, Node<T> node) {
            this.parentIdx = parentIdx;
            this.curIdx = curIdx;
            this.node = node;
        }
    }

    static abstract class NodeBuilder {
        abstract int getParentIdx();
        abstract int getCurIdx();
        abstract Node<?> build();
    }
}
