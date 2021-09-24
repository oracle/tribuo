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
import org.tribuo.regression.ImmutableRegressionInfo;
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
import java.util.SplittableRandom;
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

    private transient ArrayList<TreeFeature> data;

    private final ImmutableOutputInfo<Regressor> labelIDMap;

    private final ImmutableFeatureMap featureIDMap;

    private final RegressorImpurity impurity;

    private final int[] indices;

    private final float[] targets;

    private final float[] weights;

    private final String dimName;

    private final float weightSum;

    /**
     * Constructs a tree training node for regression problems.
     * @param impurity The impurity function.
     * @param tuple The data tuple.
     * @param dimIndex The output dimension index of this node.
     * @param dimName The output dimension name.
     * @param numExamples The number of examples.
     * @param featureIDMap The feature domain.
     * @param outputInfo The output domain.
     * @param leafDeterminer The leaf determination parameters.
     */
    public RegressorTrainingNode(RegressorImpurity impurity, InvertedData tuple, int dimIndex, String dimName,
                                 int numExamples, ImmutableFeatureMap featureIDMap,
                                 ImmutableOutputInfo<Regressor> outputInfo, LeafDeterminer leafDeterminer) {
        this(impurity,tuple.copyData(),tuple.indices,tuple.targets[dimIndex],tuple.weights,dimName,numExamples,0,
                featureIDMap,outputInfo, leafDeterminer);
    }

    private RegressorTrainingNode(RegressorImpurity impurity, ArrayList<TreeFeature> data, int[] indices,
                                  float[] targets, float[] weights, String dimName, int numExamples, int depth,
                                  ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> labelIDMap,
                                  LeafDeterminer leafDeterminer) {
        super(depth, numExamples, leafDeterminer);
        this.data = data;
        this.featureIDMap = featureIDMap;
        this.labelIDMap = labelIDMap;
        this.impurity = impurity;
        this.indices = indices;
        this.targets = targets;
        this.weights = weights;
        this.dimName = dimName;
        this.weightSum = Util.sum(indices,indices.length,weights);
        this.impurityScore = impurity.impurity(indices, targets, weights);
    }

    private RegressorTrainingNode(RegressorImpurity impurity, ArrayList<TreeFeature> data, int[] indices,
                                  float[] targets, float[] weights, String dimName, int numExamples, int depth,
                                  ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> labelIDMap,
                                  LeafDeterminer leafDeterminer, float weightSum, double impurityScore) {
        super(depth, numExamples, leafDeterminer);
        this.data = data;
        this.featureIDMap = featureIDMap;
        this.labelIDMap = labelIDMap;
        this.impurity = impurity;
        this.indices = indices;
        this.targets = targets;
        this.weights = weights;
        this.dimName = dimName;
        this.weightSum = weightSum;
        this.impurityScore = impurityScore;
    }

    @Override
    public double getImpurity() {
        return impurityScore;
    }

    @Override
    public float getWeightSum() {
        return weightSum;
    }

    /**
     * Builds a tree according to CART (as it does not do multi-way splits on categorical values like C4.5).
     * @param featureIDs Indices of the features available in this split.
     * @param rng Splittable random number generator.
     * @param useRandomSplitPoints Whether to choose split points for features at random.
     * @return A possibly empty list of TrainingNodes.
     */
    @Override
    public List<AbstractTrainingNode<Regressor>> buildTree(int[] featureIDs, SplittableRandom rng,
                                                           boolean useRandomSplitPoints) {
        if (useRandomSplitPoints) {
            return buildRandomTree(featureIDs, rng);
        } else {
            return buildGreedyTree(featureIDs);
        }
    }

    /**
     * Builds a tree according to CART
     * @param featureIDs Indices of the features available in this split.
     * @return A possibly empty list of TrainingNodes.
     */
    private List<AbstractTrainingNode<Regressor>> buildGreedyTree(int[] featureIDs) {
        int bestID = -1;
        double bestSplitValue = 0.0;
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
        double impurityDecrease = weightSum * (getImpurity() - bestScore);
        // If we found a split better than the current impurity.
        if ((bestID != -1) && (impurityDecrease >= leafDeterminer.getScaledMinImpurityDecrease())) {
            output = splitAtBest(featureIDs, bestID, bestSplitValue, bestLeftIndices, bestRightIndices);
        } else {
            output = Collections.emptyList();
        }
        data = null;
        return output;
    }

    /**
     * Builds a CART tree with randomly chosen split points.
     * @param featureIDs Indices of the features available in this split.
     * @param rng Splittable random number generator.
     * @return A possibly empty list of TrainingNodes.
     */
    private List<AbstractTrainingNode<Regressor>> buildRandomTree(int[] featureIDs, SplittableRandom rng) {
        int bestID = -1;
        double bestSplitValue = 0.0;
        double bestScore = getImpurity();
        //logger.info("Cur node score = " + bestScore);
        List<int[]> curLeftIndices = new ArrayList<>();
        List<int[]> curRightIndices = new ArrayList<>();
        List<int[]> bestLeftIndices = new ArrayList<>();
        List<int[]> bestRightIndices = new ArrayList<>();

        // split each feature once randomly and record the least impure amongst these
        for (int i = 0; i < featureIDs.length; i++) {
            List<InvertedFeature> feature = data.get(featureIDs[i]).getFeature();
            // if there is only 1 inverted feature for this feature, it has only 1 value, so cannot be split
            if (feature.size() == 1) {
                continue;
            }

            int splitIdx = rng.nextInt(feature.size()-1);

            for (int j = 0; j < splitIdx + 1; j++) {
                InvertedFeature vf;
                vf = feature.get(j);
                curLeftIndices.add(vf.indices());
            }
            for (int j = splitIdx + 1; j < feature.size(); j++) {
                InvertedFeature vf;
                vf = feature.get(j);
                curRightIndices.add(vf.indices());
            }

            ImpurityTuple lessThanScore = impurity.impurityTuple(curLeftIndices,targets,weights);
            ImpurityTuple greaterThanScore = impurity.impurityTuple(curRightIndices,targets,weights);
            double score = (lessThanScore.impurity*lessThanScore.weight + greaterThanScore.impurity*greaterThanScore.weight) / weightSum;
            if (score < bestScore) {
                bestID = i;
                bestScore = score;
                bestSplitValue = (feature.get(splitIdx).value + feature.get(splitIdx + 1).value) / 2.0;
                // Clear out the old best indices before storing the new ones.
                bestLeftIndices.clear();
                bestLeftIndices.addAll(curLeftIndices);
                bestRightIndices.clear();
                bestRightIndices.addAll(curRightIndices);
                //logger.info("id = " + featureIDs[i] + ", split = " + bestSplitValue + ", score = " + score);
                //logger.info("less score = " +lessThanScore+", less size = "+lessThanIndices.size+", greater score = " + greaterThanScore+", greater size = "+greaterThanIndices.size);
            }
        }

        List<AbstractTrainingNode<Regressor>> output;
        double impurityDecrease = weightSum * (getImpurity() - bestScore);
        // If we found a split better than the current impurity.
        if ((bestID != -1) && (impurityDecrease >= leafDeterminer.getScaledMinImpurityDecrease())) {
            output = splitAtBest(featureIDs, bestID, bestSplitValue, bestLeftIndices, bestRightIndices);
        } else {
            output = Collections.emptyList();
        }
        data = null;
        return output;
    }

    /**
     * Splits the data to form two nodes.
     * @param featureIDs Indices of the features available in this split.
     * @param bestID ID of the feature on which the split should be based.
     * @param bestSplitValue Feature value to use for splitting the data.
     * @param bestLeftIndices The indices of the examples less than or equal to the split value for the given feature.
     * @param bestRightIndices The indices of the examples greater than the split value for the given feature.
     * @return A list of training nodes resulting from the split.
     */
    private List<AbstractTrainingNode<Regressor>> splitAtBest(int[] featureIDs, int bestID, double bestSplitValue,
                                                             List<int[]> bestLeftIndices, List<int[]> bestRightIndices){

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

        float leftWeightSum = Util.sum(leftIndices,leftIndices.length,weights);
        double leftImpurityScore = impurity.impurity(leftIndices, targets, weights);

        float rightWeightSum = Util.sum(rightIndices,rightIndices.length,weights);
        double rightImpurityScore = impurity.impurity(rightIndices, targets, weights);

        boolean shouldMakeLeftLeaf = shouldMakeLeaf(leftImpurityScore, leftWeightSum);
        boolean shouldMakeRightLeaf = shouldMakeLeaf(rightImpurityScore, rightWeightSum);

        if (shouldMakeLeftLeaf && shouldMakeRightLeaf) {
            lessThanOrEqual = createLeaf(leftImpurityScore, leftIndices);
            greaterThan = createLeaf(rightImpurityScore, rightIndices);
            return Collections.emptyList();
        }

        //logger.info("Splitting on feature " + bestID + " with value " + bestSplitValue + " at depth " + depth + ", " + numExamples + " examples in node.");
        //logger.info("left indices length = " + leftIndices.length);
        ArrayList<TreeFeature> lessThanData = new ArrayList<>(data.size());
        ArrayList<TreeFeature> greaterThanData = new ArrayList<>(data.size());
        for (TreeFeature feature : data) {
            Pair<TreeFeature,TreeFeature> split = feature.split(leftIndices, rightIndices, firstBuffer, secondBuffer);
            lessThanData.add(split.getA());
            greaterThanData.add(split.getB());
        }

        List<AbstractTrainingNode<Regressor>> output = new ArrayList<>(2);
        AbstractTrainingNode<Regressor> tmpNode;
        if (shouldMakeLeftLeaf) {
            lessThanOrEqual = createLeaf(leftImpurityScore, leftIndices);
        } else {
            tmpNode = new RegressorTrainingNode(impurity, lessThanData, leftIndices, targets, weights, dimName,
                    leftIndices.length, depth + 1, featureIDMap, labelIDMap, leafDeterminer, leftWeightSum, leftImpurityScore);
            lessThanOrEqual = tmpNode;
            output.add(tmpNode);
        }

        if (shouldMakeRightLeaf) {
            greaterThan = createLeaf(rightImpurityScore, rightIndices);
        } else {
            tmpNode = new RegressorTrainingNode(impurity, greaterThanData, rightIndices, targets, weights, dimName,
                    rightIndices.length, depth + 1, featureIDMap, labelIDMap, leafDeterminer, rightWeightSum, rightImpurityScore);
            greaterThan = tmpNode;
            output.add(tmpNode);
        }
        return output;
    }

    /**
     * Generates a test time tree (made of {@link SplitNode} and {@link LeafNode}) from the tree rooted at this node.
     * @return A subtree using the SplitNode and LeafNode classes.
     */
    @Override
    public Node<Regressor> convertTree() {
        if (split) {
            return createSplitNode();
        } else {
            return createLeaf(getImpurity(), indices);
        }
    }

    /**
     * Makes a {@link LeafNode}
     * @param impurityScore the impurity score for the node.
     * @param leafIndices the indices of the examples to be placed in the node.
     * @return A {@link LeafNode}
     */
    private LeafNode<Regressor> createLeaf(double impurityScore, int[] leafIndices) {
        double mean = 0.0;
        double leafWeightSum = 0.0;
        double variance = 0.0;
        for (int i = 0; i < leafIndices.length; i++) {
            int idx = leafIndices[i];
            float value = targets[idx];
            float weight = weights[idx];

            leafWeightSum += weight;
            double oldMean = mean;
            mean += (weight / leafWeightSum) * (value - oldMean);
            variance += weight * (value - oldMean) * (value - mean);
        }
        variance = leafIndices.length > 1 ? variance / (leafWeightSum-1) : 0;
        DimensionTuple leafPred = new DimensionTuple(dimName,mean,variance);
        return new LeafNode<>(impurityScore,leafPred,Collections.emptyMap(),false);
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

        int[] ids = ((ImmutableRegressionInfo) labelInfo).getNaturalOrderToIDMapping();
        for (int i = 0; i < examples.size(); i++) {
            Example<Regressor> e = examples.getExample(i);
            indices[i] = i;
            weights[i] = e.getWeight();
            double[] output = e.getOutput().getValues();
            for (int j = 0; j < output.length; j++) {
                targets[ids[j]][i] = (float) output[j];
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