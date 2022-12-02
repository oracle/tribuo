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

package org.tribuo.classification.dtree.impl;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.dtree.impurity.LabelImpurity;
import org.tribuo.common.tree.AbstractTrainingNode;
import org.tribuo.common.tree.LeafNode;
import org.tribuo.common.tree.Node;
import org.tribuo.common.tree.SplitNode;
import org.tribuo.common.tree.impl.IntArrayContainer;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.util.Util;

import java.io.IOException;
import java.io.NotSerializableException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * A decision tree node used at training time.
 * Contains a list of the example indices currently found in this node,
 * the current impurity and a bunch of other statistics.
 */
public class ClassifierTrainingNode extends AbstractTrainingNode<Label> {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(ClassifierTrainingNode.class.getName());

    private static final ThreadLocal<IntArrayContainer> mergeBufferOne = ThreadLocal.withInitial(() -> new IntArrayContainer(DEFAULT_SIZE));
    private static final ThreadLocal<IntArrayContainer> mergeBufferTwo = ThreadLocal.withInitial(() -> new IntArrayContainer(DEFAULT_SIZE));
    private static final ThreadLocal<IntArrayContainer> mergeBufferThree = ThreadLocal.withInitial(() -> new IntArrayContainer(DEFAULT_SIZE));

    private transient ArrayList<TreeFeature> data;

    private final ImmutableOutputInfo<Label> labelIDMap;

    private final ImmutableFeatureMap featureIDMap;

    private final LabelImpurity impurity;

    private final float[] weightedLabelCounts;

    private final float weightSum;

    /**
     * Constructor which creates the inverted file.
     * @param impurity The impurity function to use.
     * @param examples The training data.
     * @param leafDeterminer Contains parameters needed to determine whether a child node will be a leaf.
     */
    public ClassifierTrainingNode(LabelImpurity impurity, Dataset<Label> examples, LeafDeterminer leafDeterminer) {
        this(impurity,invertData(examples), examples.size(), 0, examples.getFeatureIDMap(),
                examples.getOutputIDInfo(),leafDeterminer);
    }

    private ClassifierTrainingNode(LabelImpurity impurity, ArrayList<TreeFeature> data, int numExamples, int depth,
                                   ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap,
                                   LeafDeterminer leafDeterminer) {
        super(depth, numExamples, leafDeterminer);
        this.data = data;
        this.featureIDMap = featureIDMap;
        this.labelIDMap = labelIDMap;
        this.impurity = impurity;
        this.weightedLabelCounts = data.get(0).getWeightedLabelCounts();
        this.weightSum = Util.sum(weightedLabelCounts);
        this.impurityScore = impurity.impurity(weightedLabelCounts);
    }

    private ClassifierTrainingNode(LabelImpurity impurity, ArrayList<TreeFeature> data, int numExamples, int depth,
                                   ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap,
                                   LeafDeterminer leafDeterminer, float[] weightedLabelCounts, float weightSum,
                                   double impurityScore) {
        super(depth, numExamples, leafDeterminer);
        this.data = data;
        this.featureIDMap = featureIDMap;
        this.labelIDMap = labelIDMap;
        this.impurity = impurity;
        this.weightedLabelCounts = weightedLabelCounts;
        this.weightSum = weightSum;
        this.impurityScore = impurityScore;
    }

    @Override
    public float getWeightSum() {
        return weightSum;
    }

    @Override
    public double getImpurity() {
        return impurityScore;
    }

    /**
     * Builds a tree according to CART (as it does not do multi-way splits on categorical values like C4.5).
     * @param featureIDs Indices of the features available in this split.
     * @param rng Splittable random number generator.
     * @param useRandomSplitPoints Whether to choose split points for features at random.
     * @return A possibly empty list of TrainingNodes.
     */
    @Override
    public List<AbstractTrainingNode<Label>> buildTree(int[] featureIDs, SplittableRandom rng,
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
    private List<AbstractTrainingNode<Label>> buildGreedyTree(int[] featureIDs) {
        int bestID = -1;
        double bestSplitValue = 0.0;
        double bestScore = getImpurity();
        float[] lessThanCountsOfBest = new float[weightedLabelCounts.length];
        float[] greaterThanCountsOfBest = new float[weightedLabelCounts.length];
        float[] lessThanCounts = new float[weightedLabelCounts.length];
        float[] greaterThanCounts = new float[weightedLabelCounts.length];
        for (int i = 0; i < featureIDs.length; i++) {
            List<InvertedFeature> feature = data.get(featureIDs[i]).getFeature();
            Arrays.fill(lessThanCounts,0.0f);
            System.arraycopy(weightedLabelCounts, 0, greaterThanCounts, 0, weightedLabelCounts.length);
            // searching for the intervals between features.
            for (int j = 0; j < feature.size()-1; j++) {
                InvertedFeature f = feature.get(j);
                float[] featureCounts = f.getWeightedLabelCounts();
                Util.inPlaceAdd(lessThanCounts,featureCounts);
                Util.inPlaceSubtract(greaterThanCounts,featureCounts);
                // impurityWeighted rescales the impurity by the sum, in order to average properly when you add the
                // left side & right side together, and then divide by the sum of the counts
                double lessThanScore = impurity.impurityWeighted(lessThanCounts);
                double greaterThanScore = impurity.impurityWeighted(greaterThanCounts);
                double score = (lessThanScore + greaterThanScore) / weightSum;
                if (score < bestScore) {
                    bestID = i;
                    bestScore = score;
                    System.arraycopy(lessThanCounts,0,lessThanCountsOfBest,0,lessThanCounts.length);
                    System.arraycopy(greaterThanCounts,0,greaterThanCountsOfBest,0,greaterThanCounts.length);
                    bestSplitValue = (f.value + feature.get(j + 1).value) / 2.0;
                }
            }
        }

        List<AbstractTrainingNode<Label>> output;
        double impurityDecrease = weightSum * (getImpurity() - bestScore);
        // If we found a split better than the current impurity.
        if ((bestID != -1) && (impurityDecrease >= leafDeterminer.getScaledMinImpurityDecrease())) {
            output = splitAtBest(featureIDs, bestID, bestSplitValue, lessThanCountsOfBest, greaterThanCountsOfBest);
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
    public List<AbstractTrainingNode<Label>> buildRandomTree(int[] featureIDs, SplittableRandom rng) {
        int bestID = -1;
        double bestSplitValue = 0.0;
        double bestScore = getImpurity();
        float[] lessThanCountsOfBest = new float[weightedLabelCounts.length];
        float[] greaterThanCountsOfBest = new float[weightedLabelCounts.length];
        float[] lessThanCounts = new float[weightedLabelCounts.length];
        float[] greaterThanCounts = new float[weightedLabelCounts.length];

        // split each feature once randomly and record the least impure amongst these
        for (int i = 0; i < featureIDs.length; i++) {
            List<InvertedFeature> feature = data.get(featureIDs[i]).getFeature();

            // if there is only 1 inverted feature for this feature, it has only 1 value, so cannot be split
            if (feature.size() == 1) {
                continue;
            }

            Arrays.fill(lessThanCounts,0.0f);
            System.arraycopy(weightedLabelCounts, 0, greaterThanCounts, 0, weightedLabelCounts.length);

            int splitIdx = rng.nextInt(feature.size()-1);
            for (int j = 0; j < splitIdx + 1; j++) {
                InvertedFeature vf = feature.get(j);
                float[] countsBelowOrEqual = vf.getWeightedLabelCounts();
                Util.inPlaceAdd(lessThanCounts, countsBelowOrEqual);
                Util.inPlaceSubtract(greaterThanCounts, countsBelowOrEqual);
            }
            double lessThanScore = impurity.impurityWeighted(lessThanCounts);
            double greaterThanScore = impurity.impurityWeighted(greaterThanCounts);
            double score = (lessThanScore + greaterThanScore) / weightSum;
            if (score < bestScore) {
                bestID = i;
                bestScore = score;
                System.arraycopy(lessThanCounts,0,lessThanCountsOfBest,0,lessThanCounts.length);
                System.arraycopy(greaterThanCounts,0,greaterThanCountsOfBest,0,greaterThanCounts.length);
                bestSplitValue = (feature.get(splitIdx).value + feature.get(splitIdx + 1).value) / 2.0;
            }
        }

        List<AbstractTrainingNode<Label>> output;
        double impurityDecrease = weightSum * (getImpurity() - bestScore);
        // If we found a split better than the current impurity.
        if ((bestID != -1) && (impurityDecrease >= leafDeterminer.getScaledMinImpurityDecrease())) {
            output = splitAtBest(featureIDs, bestID, bestSplitValue, lessThanCountsOfBest, greaterThanCountsOfBest);
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
     * @param lessThanCounts Weighted label counts for data less than or equal to the split value for the given feature.
     * @param greaterThanCounts Weighted label counts for data greater than the split value for the given feature.
     * @return A list of training nodes resulting from the split.
     */
    private List<AbstractTrainingNode<Label>> splitAtBest(int[] featureIDs, int bestID, double bestSplitValue,
                                                          float[] lessThanCounts, float[] greaterThanCounts) {
        splitID = featureIDs[bestID];
        split = true;
        splitValue = bestSplitValue;

        float lessThanWeightSum = Util.sum(lessThanCounts);
        double lessThanImpurityScore = impurity.impurity(lessThanCounts);

        float greaterThanWeightSum = Util.sum(greaterThanCounts);
        double greaterThanImpurityScore = impurity.impurity(greaterThanCounts);

        boolean shouldMakeLessThanLeaf = shouldMakeLeaf(lessThanImpurityScore, lessThanWeightSum);
        boolean shouldMakeGreaterThanLeaf = shouldMakeLeaf(greaterThanImpurityScore, greaterThanWeightSum);

        if (shouldMakeLessThanLeaf && shouldMakeGreaterThanLeaf) {
            lessThanOrEqual = createLeaf(lessThanImpurityScore, lessThanCounts);
            greaterThan = createLeaf(greaterThanImpurityScore, greaterThanCounts);
            return Collections.emptyList();
        }

        IntArrayContainer lessThanIndices = mergeBufferOne.get();
        lessThanIndices.size = 0;
        IntArrayContainer buffer = mergeBufferTwo.get();
        buffer.size = 0;
        for (InvertedFeature f : data.get(splitID)) {
            if (f.value < splitValue) {
                int[] indices = f.indices();
                IntArrayContainer.merge(lessThanIndices,indices,buffer);
                // Swap the buffers
                IntArrayContainer tmp = lessThanIndices;
                lessThanIndices = buffer;
                buffer = tmp;
            } else {
                break;
            }
        }
        //logger.info("Splitting on feature " + maxID + " with value " + maxSplitValue + " at depth " + depth + ", " + numExamples + " examples in node.");
        //logger.info("left indices length = " + lessThanIndices.size);
        IntArrayContainer secondBuffer = mergeBufferThree.get();
        secondBuffer.grow(lessThanIndices.size);
        ArrayList<TreeFeature> lessThanData = new ArrayList<>(data.size());
        ArrayList<TreeFeature> greaterThanData = new ArrayList<>(data.size());
        for (TreeFeature feature : data) {
            Pair<TreeFeature,TreeFeature> split = feature.split(lessThanIndices,buffer,secondBuffer);
            lessThanData.add(split.getA());
            greaterThanData.add(split.getB());
        }

        List<AbstractTrainingNode<Label>> output = new ArrayList<>(2);
        if (shouldMakeLessThanLeaf) {
            lessThanOrEqual = createLeaf(lessThanImpurityScore, lessThanCounts);
        } else {
            AbstractTrainingNode<Label> tmpNode = new ClassifierTrainingNode(impurity, lessThanData, lessThanIndices.size, depth + 1,
                    featureIDMap, labelIDMap, leafDeterminer, lessThanCounts, lessThanWeightSum, lessThanImpurityScore);
            lessThanOrEqual = tmpNode;
            output.add(tmpNode);
        }

        if (shouldMakeGreaterThanLeaf) {
            greaterThan = createLeaf(greaterThanImpurityScore, greaterThanCounts);
        } else {
            AbstractTrainingNode<Label> tmpNode = new ClassifierTrainingNode(impurity, greaterThanData, numExamples - lessThanIndices.size,
                    depth + 1, featureIDMap, labelIDMap, leafDeterminer, greaterThanCounts, greaterThanWeightSum, greaterThanImpurityScore);
            greaterThan = tmpNode;
            output.add(tmpNode);
        }
        return output;
    }

    /**
     * Makes a {@link LeafNode}
     * @param impurityScore the impurity score for the node.
     * @param weightedCounts the weighted label counts of the data in the node.
     * @return a {@link LeafNode}
     */
    private LeafNode<Label> createLeaf(double impurityScore, float[] weightedCounts) {
        double[] normedCounts = Util.normalizeToDistribution(weightedCounts);
        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String,Label> counts = new LinkedHashMap<>();
        for (int i = 0; i < weightedCounts.length; i++) {
            final double curCount = normedCounts[i];
            String name = labelIDMap.getOutput(i).getLabel();
            Label label = new Label(name,curCount);
            counts.put(name, label);
            if (curCount > maxScore) {
                maxScore = curCount;
                maxLabel = label;
            }
        }
        return new LeafNode<>(impurityScore,maxLabel,counts,true);
    }

    /**
     * Generates a test time tree (made of {@link SplitNode} and {@link LeafNode}) from the tree rooted at this node.
     * @return A subtree using the SplitNode and LeafNode classes.
     */
    @Override
    public Node<Label> convertTree() {
        if (split) {
            return createSplitNode();
        } else {
            return createLeaf(getImpurity(), weightedLabelCounts);
        }
    }

    /**
     * Inverts a training dataset from row major to column major. This partially de-sparsifies the dataset
     * so it's very expensive in terms of memory.
     * @param examples An input dataset.
     * @return A list of TreeFeatures which contain {@link InvertedFeature}s.
     */
    private static ArrayList<TreeFeature> invertData(Dataset<Label> examples) {
        ImmutableFeatureMap featureInfos = examples.getFeatureIDMap();
        ImmutableOutputInfo<Label> labelInfo = examples.getOutputIDInfo();
        int numLabels = labelInfo.size();
        int numFeatures = featureInfos.size();
        int numExamples = examples.size();

        int[] labels = new int[numExamples];
        float[] weights = new float[numExamples];

        int k = 0;
        for (Example<Label> e : examples) {
            weights[k] = e.getWeight();
            labels[k] = labelInfo.getID(e.getOutput());
            k++;
        }

        logger.fine("Building initial List<TreeFeature> for " + numFeatures + " features and " + numLabels + " classes");
        ArrayList<TreeFeature> data = new ArrayList<>(featureInfos.size());

        for (int i = 0; i < featureInfos.size(); i++) {
            data.add(new TreeFeature(i,numLabels,labels,weights));
        }

        for (int i = 0; i < examples.size(); i++) {
            Example<Label> e = examples.getExample(i);
            SparseVector vec = SparseVector.createSparseVector(e,featureInfos,false);
            int lastID = 0;
            for (VectorTuple f : vec) {
                int curID = f.index;
                for (int j = lastID; j < curID; j++) {
                    data.get(j).observeValue(0.0,i);
                }
                data.get(curID).observeValue(f.value,i);
                //
                // These two checks should never occur as SparseVector deals with collisions, and Dataset prevents
                // repeated features.
                // They are left in just to make sure.
                if (lastID > curID) {
                    logger.severe("Example = " + e);
                    throw new IllegalStateException("Features aren't ordered. At id " + i + ", lastID = " + lastID + ", curID = " + curID);
                } else if (lastID-1 == curID) {
                    logger.severe("Example = " + e);
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

        return data;
    }

    private void writeObject(java.io.ObjectOutputStream stream)
            throws IOException {
        throw new NotSerializableException("ClassifierTrainingNode is a runtime class only, and should not be serialized.");
    }
}