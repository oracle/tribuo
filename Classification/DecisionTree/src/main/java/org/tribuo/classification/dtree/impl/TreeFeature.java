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

package org.tribuo.classification.dtree.impl;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.common.tree.impl.IntArrayContainer;
import org.tribuo.util.Util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * An inverted feature, which stores a reference to all the values of this feature.
 *
 * Can be split into two values based on an example index list.
 */
class TreeFeature implements Iterable<InvertedFeature>, Serializable {

    private final int id;

    private final List<InvertedFeature> feature;

    private final Map<Double,InvertedFeature> valueMap;

    private final int numLabels;

    private final int[] labels;

    private final float[] weights;

    private boolean sorted = true;

    TreeFeature(int id, int numLabels, int[] labels, float[] weights) {
        this.id = id;
        this.numLabels = numLabels;
        this.feature = new ArrayList<>();
        this.valueMap = new HashMap<>();
        this.labels = labels;
        this.weights = weights;
    }

    /**
     * This constructor doesn't make a valueMap, and is only used when all data has been observed.
     * So it will throw NullPointerException if you call observeValue();
     * @param id The id number for this feature.
     * @param numLabels The number of labels in this feature.
     * @param data The data.
     */
    private TreeFeature(int id, int numLabels, List<InvertedFeature> data) {
        this.id = id;
        this.numLabels = numLabels;
        this.feature = data;
        this.valueMap = null;
        this.labels = null;
        this.weights = null;
    }

    @Override
    public Iterator<InvertedFeature> iterator() {
        return feature.iterator();
    }

    public List<InvertedFeature> getFeature() {
        return feature;
    }

    /**
     * Observes a value for this feature.
     * @param value The value observed.
     * @param exampleID The example id number.
     */
    public void observeValue(double value, int exampleID) {
        InvertedFeature f = valueMap.get(value);
        if (f == null) {
            f = new InvertedFeature(value,exampleID,numLabels,labels,weights);
            valueMap.put(value,f);
            feature.add(f);
            // feature list is no longer guaranteed to be sorted
            sorted = false;
        } else {
            // Update currently known feature
            f.add(exampleID);
        }
    }

    /**
     * Sort the list using InvertedFeature's natural ordering. Must be done after all elements are inserted.
     */
    public void sort() {
        feature.sort(null);
        sorted = true;
    }

    /**
     * Fixes the size of each {@link InvertedFeature}'s inner arrays.
     */
    public void fixSize() {
        feature.forEach(InvertedFeature::fixSize);
    }

    public float[] getWeightedLabelCounts() {
        float[] weightedLabelCounts = new float[numLabels];

        for (InvertedFeature f : feature) {
            Util.inPlaceAdd(weightedLabelCounts,f.getWeightedLabelCounts());
        }

        return weightedLabelCounts;
    }

    /**
     * Splits this tree feature into two.
     *
     * @param leftIndices The indices to go in the left branch.
     * @param firstBuffer A buffer to use.
     * @param secondBuffer Another buffer.
     * @return A pair of TreeFeatures, the first element is the left branch, the second the right.
     */
    public Pair<TreeFeature,TreeFeature> split(IntArrayContainer leftIndices, IntArrayContainer firstBuffer, IntArrayContainer secondBuffer) {
        if (!sorted) {
            throw new IllegalStateException("TreeFeature must be sorted before split is called");
        }
        List<InvertedFeature> leftFeatures = new ArrayList<>();
        List<InvertedFeature> rightFeatures = new ArrayList<>();

        firstBuffer.fill(leftIndices);
        for (InvertedFeature f : feature) {
            // Check if we've exhausted all the left side indices
            if (firstBuffer.size > 0) {
                Pair<InvertedFeature, InvertedFeature> split = f.split(firstBuffer, secondBuffer);
                IntArrayContainer tmp = secondBuffer;
                secondBuffer = firstBuffer;
                firstBuffer = tmp;
                InvertedFeature left = split.getA();
                InvertedFeature right = split.getB();
                if (left != null) {
                    leftFeatures.add(left);
                }
                if (right != null) {
                    rightFeatures.add(right);
                }
            } else {
                rightFeatures.add(f);
            }
        }

        return new Pair<>(new TreeFeature(id,numLabels,leftFeatures),new TreeFeature(id,numLabels,rightFeatures));
    }

    @Override
    public String toString() {
        return "TreeFeature(id="+id+",values="+feature.toString()+")";
    }
}
