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
import org.tribuo.common.tree.impl.IntArrayContainer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * An inverted feature, which stores a reference to all the values of this feature.
 * <p>
 * Can be split into two values based on an example index list.
 */
public class TreeFeature implements Iterable<InvertedFeature> {

    private final int id;

    private final List<InvertedFeature> feature;

    private final Map<Double,InvertedFeature> valueMap;

    private boolean sorted = true;

    /**
     * Constructs an inverted feature with the specified feature id.
     * @param id The feature id.
     */
    public TreeFeature(int id) {
        this.id = id;
        this.feature = new ArrayList<>();
        this.valueMap = new HashMap<>();
    }

    /**
     * This constructor doesn't make a valueMap, and is only used when all data has been observed.
     * So it will throw NullPointerException if you call observeValue();
     * @param id The id number for this feature.
     * @param data The data.
     */
    private TreeFeature(int id, List<InvertedFeature> data) {
        this.id = id;
        this.feature = data;
        this.valueMap = null;
    }

    /**
     * Constructor used by {@link TreeFeature#deepCopy}.
     * @param id The id number for this feature.
     * @param data The data.
     * @param valueMap The value map.
     * @param sorted Is this data sorted.
     */
    private TreeFeature(int id, List<InvertedFeature> data, Map<Double,InvertedFeature> valueMap, boolean sorted) {
        this.id = id;
        this.feature = data;
        this.valueMap = valueMap;
        this.sorted = sorted;
    }

    @Override
    public Iterator<InvertedFeature> iterator() {
        return feature.iterator();
    }

    /**
     * Gets the inverted feature values for this feature.
     * @return The list of feature values.
     */
    public List<InvertedFeature> getFeature() {
        return feature;
    }

    /**
     * Observes a value for this feature.
     * @param value The value observed.
     * @param exampleID The example id number.
     */
    public void observeValue(double value, int exampleID) {
        Double dValue = value;
        InvertedFeature f = valueMap.get(dValue);
        if (f == null) {
            f = new InvertedFeature(value,exampleID);
            valueMap.put(dValue,f);
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

    /**
     * Splits this tree feature into two.
     *
     * @param leftIndices The indices to go in the left branch.
     * @param rightIndices The indices to go in the right branch.
     * @param firstBuffer A buffer for temporary work.
     * @param secondBuffer A buffer for temporary work.
     * @return A pair of TreeFeatures, the first element is the left branch, the second the right.
     */
    public Pair<TreeFeature,TreeFeature> split(int[] leftIndices, int[] rightIndices, IntArrayContainer firstBuffer, IntArrayContainer secondBuffer) {
        if (!sorted) {
            throw new IllegalStateException("TreeFeature must be sorted before split is called");
        }

        List<InvertedFeature> leftFeatures;
        List<InvertedFeature> rightFeatures;
        if (feature.size() == 1) {
            double value = feature.get(0).value;
            leftFeatures = Collections.singletonList(new InvertedFeature(value,leftIndices));
            rightFeatures = Collections.singletonList(new InvertedFeature(value,rightIndices));
        } else {
            leftFeatures = new ArrayList<>();
            rightFeatures = new ArrayList<>();
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
        }

        return new Pair<>(new TreeFeature(id,leftFeatures),new TreeFeature(id,rightFeatures));

    }

    public String toString() {
        return "TreeFeature(id="+id+",values="+feature.toString()+")";
    }

    /**
     * Returns a deep copy of this tree feature.
     * @return A deep copy.
     */
    public TreeFeature deepCopy() {
        Map<Double,InvertedFeature> newValueMap;
        List<InvertedFeature> newFeature = new ArrayList<>();
        if (valueMap != null) {
            newValueMap = new HashMap<>();
            for (Map.Entry<Double,InvertedFeature> e : valueMap.entrySet()) {
                InvertedFeature featureCopy = e.getValue().deepCopy();
                newValueMap.put(e.getKey(),featureCopy);
                newFeature.add(featureCopy);
                newFeature.sort(null);
            }
        } else {
            newValueMap = null;
            for (InvertedFeature f : feature) {
                newFeature.add(f.deepCopy());
            }
        }
        return new TreeFeature(id,newFeature,newValueMap,true);
    }
}
