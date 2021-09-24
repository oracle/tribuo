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

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

/**
 * Internal datastructure for implementing a decision tree.
 * <p>
 * Represents a single value and feature tuple, with associated arrays for
 * storing how many examples and labels have that value.
 * <p>
 * Indices and values must be inserted in sorted ascending order or everything will break.
 * This code does not check that this invariant is maintained.
 * <p>
 * Note: this class has a natural ordering that is inconsistent with equals.
 * While this class is serializable, it should not be serialized in most cases, as it's only used during training.
 */
class InvertedFeature implements Comparable<InvertedFeature>, Serializable {

    private static final int DEFAULT_SIZE = 8;

    /**
     * The value of this feature.
     */
    public final double value;

    private final int numLabels;
    private final float[] weightedLabelCounts;

    /**
     * indices *must* be inserted in a sorted order.
     */
    private int[] indices = null;
    private int curSize = -1;

    /**
     * labels and weights are immutable and are shared across the whole dataset.
     */
    private final int[] labels;
    private final float[] weights;

    /**
     * This is a short circuit in case there is a single index in this feature.
     */
    private int index;

    InvertedFeature(double value, int[] indices, int numLabels, int[] labels, float[] weights) {
        this.value = value;
        this.indices = indices;
        this.labels = labels;
        this.weights = weights;
        this.curSize = indices.length;
        this.numLabels = numLabels;
        this.weightedLabelCounts = new float[numLabels];
        for (int idx : indices) {
            weightedLabelCounts[labels[idx]] += weights[idx];
        }
    }

    InvertedFeature(double value, int index, int numLabels, int[] labels, float[] weights) {
        this.value = value;
        this.index = index;
        this.numLabels = numLabels;
        this.labels = labels;
        this.weights = weights;
        this.weightedLabelCounts = new float[numLabels];
        this.weightedLabelCounts[labels[index]] += weights[index];
    }

    public void add(int index) {
        if (indices == null) {
            initArrays();
        }
        append(index);
        weightedLabelCounts[labels[index]] += weights[index];
    }

    private void append(int index) {
        if (curSize == indices.length) {
            // growth strategy is from ArrayList, if it's good enough for that it's good enough for me.
            int newSize = indices.length + (indices.length >> 1);
            indices = Arrays.copyOf(indices,newSize);
        }
        indices[curSize] = index;
        curSize++;
    }

    public int[] indices() {
        if (indices != null) {
            return indices;
        } else {
            int[] ret = new int[1];
            ret[0] = index;
            return ret;
        }
    }

    public float[] getWeightedLabelCounts() {
        return weightedLabelCounts;
    }

    public void fixSize() {
        if (indices != null) {
            indices = Arrays.copyOf(indices, curSize);
        }
    }

    /**
     * Relies upon allLeftIndices being sorted in ascending order. Undefined when it's not.
     * @param allLeftIndices The indices of the left branch.
     * @param buffer The buffer to write out the unused indices to.
     * @return A pair, with the first element the left branch and the second element the right branch.
     */
    public Pair<InvertedFeature,InvertedFeature> split(IntArrayContainer allLeftIndices, IntArrayContainer buffer) {
        int[] allLeftArray = allLeftIndices.array;
        int allLeftSize = allLeftIndices.size;
        int[] bufferArray = buffer.array;
        if (indices != null) {
            // These are init'd to indices.length as allLeftIndices may contain indices not in this InvertedFeature.
            int[] leftIndices = new int[indices.length];
            int leftSize = 0;
            int[] rightIndices = new int[indices.length];
            int rightSize = 0;

            int bufferIdx = 0;
            int curIndex = 0;
            int j = 0;
            for (int i = 0; i < curSize; i++) {
                //relying on the shortcut evaluation so we don't pop out of allLeftArray
                while ((j < allLeftSize) && ((curIndex = allLeftArray[j]) < indices[i])) {
                    bufferArray[bufferIdx] = curIndex;
                    bufferIdx++;
                    j++;
                }
                if ((j < allLeftSize) && (allLeftArray[j] == indices[i])) {
                    //in the left indices, put in left array
                    leftIndices[leftSize] = indices[i];
                    leftSize++;
                    j++; // consume the value in allLeftIndices[j]
                } else {
                    //allLeftIndices[j] now greater than indices[i], so must not include it
                    //put in right array.
                    rightIndices[rightSize] = indices[i];
                    rightSize++;
                }
            }

            if (j < allLeftSize) {
                System.arraycopy(allLeftArray, j, bufferArray, bufferIdx, allLeftSize - j);
            }
            buffer.size = bufferIdx + (allLeftSize - j);
            allLeftIndices.size = 0;

            InvertedFeature left, right;
            if (leftSize == 0) {
                left = null;
            } else if (leftSize == 1) {
                left = new InvertedFeature(value,leftIndices[0],numLabels,labels,weights);
            } else {
                left = new InvertedFeature(value, Arrays.copyOf(leftIndices, leftSize), numLabels, labels, weights);
            }
            if (rightSize == 0) {
                right = null;
            } else if (rightSize == 1) {
                right = new InvertedFeature(value,rightIndices[0],numLabels,labels,weights);
            } else {
                right = new InvertedFeature(value, Arrays.copyOf(rightIndices, rightSize), numLabels, labels, weights);
            }
            return new Pair<>(left,right);
        } else {
            //In this case this inverted feature only holds one value, so check for it in left indices
            boolean found = false;
            int i = 0;
            while (!found && i < allLeftSize) {
                if (allLeftArray[i] == index) {
                    found = true;
                } else {
                    i++;
                }
            }
            if (found) {
                System.arraycopy(allLeftArray,0,bufferArray,0,i);
                i++;
                while (i < allLeftSize) {
                    bufferArray[i-1] = allLeftArray[i];
                    i++;
                }
                if (i < allLeftSize-1) {
                    System.arraycopy(allLeftArray, i + 1, bufferArray, i, allLeftSize - i);
                }
                buffer.size = allLeftSize-1;
                allLeftIndices.size = 0;
                return new Pair<>(new InvertedFeature(value,index,numLabels,labels,weights),null);
            } else {
                allLeftIndices.array = bufferArray;
                allLeftIndices.size = 0;
                buffer.array = allLeftArray;
                buffer.size = allLeftSize;
                return new Pair<>(null,new InvertedFeature(value,index,numLabels,labels,weights));
            }
        }
    }

    private void initArrays() {
        indices = new int[DEFAULT_SIZE];
        indices[0] = index;
        curSize = 1;
    }

    @Override
    public int compareTo(InvertedFeature o) {
        return Double.compare(value, o.value);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof InvertedFeature)) return false;
        InvertedFeature that = (InvertedFeature) o;
        return Double.compare(that.value, value) == 0 &&
                numLabels == that.numLabels &&
                curSize == that.curSize &&
                index == that.index &&
                Arrays.equals(weightedLabelCounts, that.weightedLabelCounts) &&
                Arrays.equals(indices, that.indices) &&
                Arrays.equals(labels, that.labels) &&
                Arrays.equals(weights, that.weights);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(value, numLabels, curSize, index);
        result = 31 * result + Arrays.hashCode(weightedLabelCounts);
        result = 31 * result + Arrays.hashCode(indices);
        result = 31 * result + Arrays.hashCode(labels);
        result = 31 * result + Arrays.hashCode(weights);
        return result;
    }

    @Override
    public String toString() {
        if (indices != null) {
            return "InvertedFeature(value=" + value + ",size=" + curSize + ",indices=" + Arrays.toString(indices) + ")";
        } else {
            return "InvertedFeature(value=" + value + ",size=1" + ",index=" + index + ")";
        }
    }
}
