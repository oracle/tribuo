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

package org.tribuo.regression.rtree.impl;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.common.tree.AbstractTrainingNode;
import org.tribuo.common.tree.LeafNode;
import org.tribuo.common.tree.Node;
import org.tribuo.common.tree.SplitNode;
import org.tribuo.common.tree.impl.IntArrayContainer;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.Regressor.DimensionTuple;
import org.tribuo.regression.rtree.impurity.RegressorImpurity;
import org.tribuo.regression.rtree.impurity.RegressorImpurity.ImpurityTuple;
import org.tribuo.util.Util;

import java.io.IOException;
import java.io.NotSerializableException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * A decision tree node used at training time.
 * Contains a list of the example indices currently found in this node,
 * the current impurity and a bunch of other statistics.
 */
public class RegressorTrainingNode extends AbstractTrainingNode<Regressor> {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(RegressorTrainingNode.class.getName());

    private static final ThreadLocal<IntArrayContainer> mergeBufferOne = ThreadLocal.withInitial(() -> new IntArrayContainer(DEFAULT_SIZE));
    private static final ThreadLocal<IntArrayContainer> mergeBufferTwo = ThreadLocal.withInitial(() -> new IntArrayContainer(DEFAULT_SIZE));
    private static final ThreadLocal<IntArrayContainer> mergeBufferThree = ThreadLocal.withInitial(() -> new IntArrayContainer(DEFAULT_SIZE));
    private static final ThreadLocal<IntArrayContainer> mergeBufferFour = ThreadLocal.withInitial(() -> new IntArrayContainer(DEFAULT_SIZE));
    private static final ThreadLocal<IntArrayContainer> mergeBufferFive = ThreadLocal.withInitial(() -> new IntArrayContainer(DEFAULT_SIZE));

    private transient ArrayList<TreeFeature> data;

    private final ImmutableOutputInfo<Regressor> labelIDMap;

    private final ImmutableFeatureMap featureIDMap;

    private final RegressorImpurity impurity;

    private final int[] indices;

    private final float[] targets;

    private final float[] weights;

    private final String dimName;

    public RegressorTrainingNode(RegressorImpurity impurity, InvertedData tuple, int dimIndex, String dimName, int numExamples, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputInfo) {
        this(impurity,tuple.copyData(),tuple.indices,tuple.targets[dimIndex],tuple.weights,dimName,numExamples,0,featureIDMap,outputInfo);
    }

    private RegressorTrainingNode(RegressorImpurity impurity, ArrayList<TreeFeature> data, int[] indices, float[] targets, float[] weights, String dimName, int numExamples, int depth, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> labelIDMap) {
        super(depth, numExamples);
        this.data = data;
        this.featureIDMap = featureIDMap;
        this.labelIDMap = labelIDMap;
        this.impurity = impurity;
        this.indices = indices;
        this.targets = targets;
        this.weights = weights;
        this.dimName = dimName;
    }

    @Override
    public double getImpurity() {
        return impurity.impurity(indices, targets, weights);
    }

    /**
     * Builds a tree according to CART (as it does not do multi-way splits on categorical values like C4.5).
     * @param featureIDs Indices of the features available in this split.
     * @return A possibly empty list of TrainingNodes.
     */
    @Override
    public List<AbstractTrainingNode<Regressor>> buildTree(int[] featureIDs) {
        int bestID = -1;
        double bestSplitValue = 0.0;
        double weightSum = Util.sum(indices,indices.length,weights);
        double bestScore = getImpurity();
        //logger.info("Cur node score = " + bestScore);
        List<int[]> curIndices = new ArrayList<>();
        List<int[]> bestLeftIndices = new ArrayList<>();
        List<int[]> bestRightIndices = new ArrayList<>();
        for (int i = 0; i < featureIDs.length; i++) {
            List<InvertedFeature> feature = data.get(featureIDs[i]).getFeature();

            curIndices.clear();
            for (int j = 0; j < feature.size(); j++) {
                InvertedFeature f = feature.get(j);
                int[] curFeatureIndices = f.indices();
                curIndices.add(curFeatureIndices);
            }

            // searching for the intervals between features.
            for (int j = 0; j < feature.size()-1; j++) {
                List<int[]> curLeftIndices = curIndices.subList(0,j+1);
                List<int[]> curRightIndices = curIndices.subList(j+1,feature.size());
                ImpurityTuple lessThanScore = impurity.impurityTuple(curLeftIndices,targets,weights);
                ImpurityTuple greaterThanScore = impurity.impurityTuple(curRightIndices,targets,weights);
                double score = (lessThanScore.impurity*lessThanScore.weight + greaterThanScore.impurity*greaterThanScore.weight) / weightSum;
                if (score < bestScore) {
                    bestID = i;
                    bestScore = score;
                    bestSplitValue = (feature.get(j).value + feature.get(j + 1).value) / 2.0;
                    // Clear out the old best indices before storing the new ones.
                    bestLeftIndices.clear();
                    bestLeftIndices.addAll(curLeftIndices);
                    bestRightIndices.clear();
                    bestRightIndices.addAll(curRightIndices);
                    //logger.info("id = " + featureIDs[i] + ", split = " + bestSplitValue + ", score = " + score);
                    //logger.info("less score = " +lessThanScore+", less size = "+lessThanIndices.size+", greater score = " + greaterThanScore+", greater size = "+greaterThanIndices.size);
                }
            }
        }
        List<AbstractTrainingNode<Regressor>> output;
        // If we found a split better than the current impurity.
        if (bestID != -1) {
            splitID = featureIDs[bestID];
            split = true;
            splitValue = bestSplitValue;
            IntArrayContainer firstBuffer = mergeBufferOne.get();
            firstBuffer.size = 0;
            firstBuffer.grow(indices.length);
            IntArrayContainer secondBuffer = mergeBufferTwo.get();
            secondBuffer.size = 0;
            secondBuffer.grow(indices.length);
            int[] leftIndices = IntArrayContainer.merge(bestLeftIndices, firstBuffer, secondBuffer);
            int[] rightIndices = IntArrayContainer.merge(bestRightIndices, firstBuffer, secondBuffer);
            //logger.info("Splitting on feature " + bestID + " with value " + bestSplitValue + " at depth " + depth + ", " + numExamples + " examples in node.");
            //logger.info("left indices length = " + leftIndices.length);
            ArrayList<TreeFeature> lessThanData = new ArrayList<>(data.size());
            ArrayList<TreeFeature> greaterThanData = new ArrayList<>(data.size());
            for (TreeFeature feature : data) {
                Pair<TreeFeature,TreeFeature> split = feature.split(leftIndices, rightIndices, firstBuffer, secondBuffer);
                lessThanData.add(split.getA());
                greaterThanData.add(split.getB());
            }
            lessThanOrEqual = new RegressorTrainingNode(impurity, lessThanData, leftIndices, targets, weights, dimName, leftIndices.length, depth + 1, featureIDMap, labelIDMap);
            greaterThan = new RegressorTrainingNode(impurity, greaterThanData, rightIndices, targets, weights, dimName, rightIndices.length, depth + 1, featureIDMap, labelIDMap);
            output = new ArrayList<>();
            output.add(lessThanOrEqual);
            output.add(greaterThan);
        } else {
            output = Collections.emptyList();
        }
        data = null;
        return output;
    }

    /**
     * Generates a test time tree (made of {@link SplitNode} and {@link LeafNode}) from the tree rooted at this node.
     * @return A subtree using the SplitNode and LeafNode classes.
     */
    @Override
    public Node<Regressor> convertTree() {
        if (split) {
            // split node
            Node<Regressor> newGreaterThan = greaterThan.convertTree();
            Node<Regressor> newLessThan = lessThanOrEqual.convertTree();
            return new SplitNode<>(splitValue,splitID,getImpurity(),newGreaterThan,newLessThan);
        } else {
            double mean = 0.0;
            double weightSum = 0.0;
            double variance = 0.0;
            for (int i = 0; i < indices.length; i++) {
                int idx = indices[i];
                float value = targets[idx];
                float weight = weights[idx];

                weightSum += weight;
                double oldMean = mean;
                mean += (weight / weightSum) * (value - oldMean);
                variance += weight * (value - oldMean) * (value - mean);
            }
            variance = indices.length > 1 ? variance / (weightSum-1) : 0;
            DimensionTuple leafPred = new DimensionTuple(dimName,mean,variance);
            return new LeafNode<>(getImpurity(),leafPred,Collections.emptyMap(),false);
        }
    }

    /**
     * Inverts a training dataset from row major to column major. This partially de-sparsifies the dataset
     * so it's very expensive in terms of memory.
     * @param examples An input dataset.
     * @return A list of TreeFeatures which contain {@link InvertedFeature}s.
     */
    public static InvertedData invertData(Dataset<Regressor> examples) {
        ImmutableFeatureMap featureInfos = examples.getFeatureIDMap();
        ImmutableOutputInfo<Regressor> labelInfo = examples.getOutputIDInfo();
        int numLabels = labelInfo.size();
        int numFeatures = featureInfos.size();
        int[] indices = new int[examples.size()];
        float[][] targets = new float[numLabels][examples.size()];
        float[] weights = new float[examples.size()];

        logger.fine("Building initial List<TreeFeature> for " + numFeatures + " features and " + numLabels + " outputs");
        ArrayList<TreeFeature> data = new ArrayList<>(featureInfos.size());

        for (int i = 0; i < featureInfos.size(); i++) {
            data.add(new TreeFeature(i));
        }

        for (int i = 0; i < examples.size(); i++) {
            Example<Regressor> e = examples.getExample(i);
            indices[i] = i;
            weights[i] = e.getWeight();
            double[] output = e.getOutput().getValues();
            for (int j = 0; j < output.length; j++) {
                targets[j][i] = (float) output[j];
            }
            SparseVector vec = SparseVector.createSparseVector(e,featureInfos,false);
            int lastID = 0;
            for (VectorTuple f : vec) {
                int curID = f.index;
                for (int j = lastID; j < curID; j++) {
                    data.get(j).observeValue(0.0,i);
                }
                data.get(curID).observeValue(f.value,i);
                //
                // These two checks should never occur as SparseVector deals with
                // collisions, and Dataset prevents repeated features.
                // They are left in just to make sure.
                if (lastID > curID) {
                    logger.severe("Example = " + e.toString());
                    throw new IllegalStateException("Features aren't ordered. At id " + i + ", lastID = " + lastID + ", curID = " + curID);
                } else if (lastID-1 == curID) {
                    logger.severe("Example = " + e.toString());
                    throw new IllegalStateException("Features are repeated. At id " + i + ", lastID = " + lastID + ", curID = " + curID);
                }
                lastID = curID + 1;
            }
            for (int j = lastID; j < numFeatures; j++) {
                data.get(j).observeValue(0.0,i);
            }
            if (i % 1000 == 0) {
                logger.fine("Processed example " + i);
            }
        }

        logger.fine("Sorting features");

        data.forEach(TreeFeature::sort);

        logger.fine("Fixing InvertedFeature sizes");

        data.forEach(TreeFeature::fixSize);

        logger.fine("Built initial List<TreeFeature>");

        return new InvertedData(data,indices,targets,weights);
    }

    /**
     * Tuple containing an inverted dataset (i.e., feature-wise not exmaple-wise).
     */
    public static class InvertedData {
        final ArrayList<TreeFeature> data;
        final int[] indices;
        final float[][] targets;
        final float[] weights;

        InvertedData(ArrayList<TreeFeature> data, int[] indices, float[][] targets, float[] weights) {
            this.data = data;
            this.indices = indices;
            this.targets = targets;
            this.weights = weights;
        }

        ArrayList<TreeFeature> copyData() {
            ArrayList<TreeFeature> newData = new ArrayList<>();

            for (TreeFeature f : data) {
                newData.add(f.deepCopy());
            }

            return newData;
        }
    }

    private void writeObject(java.io.ObjectOutputStream stream)
            throws IOException {
        throw new NotSerializableException("RegressorTrainingNode is a runtime class only, and should not be serialized.");
    }
}