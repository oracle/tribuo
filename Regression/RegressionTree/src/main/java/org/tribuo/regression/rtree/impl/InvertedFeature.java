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

import java.util.Arrays;

/**
 * Internal datastructure for implementing a decision tree.
 * <p>
 * Represents a single value and feature tuple, with associated arrays for
 * the indicies where that combination occurs.
 * <p>
 * Indices and values must be inserted in sorted ascending order or everything will break.
 * This code does not check that this invariant is maintained.
 * <p>
 * Note: this class has a natural ordering that is inconsistent with equals.
 */
public class InvertedFeature implements Comparable<InvertedFeature> {

    private static final int DEFAULT_SIZE = 8;

    /**
     * The feature value of this object.
     */
    public final double value;

    /**
     * Indices must be inserted in a sorted order.
     */
    private int[] indices = null;
    private int curSize = -1;

    /**
     * This is a short circuit in case there is a single index in this feature.
     */
    private int index;

    /**
     * Constructs an inverted feature for the specified value which occurs at the specified indices.
     * @param value The value.
     * @param indices The indices where the value occurs.
     */
    public InvertedFeature(double value, int[] indices) {
        this.value = value;
        this.indices = indices;
        this.curSize = indices.length;
    }

    /**
     * Constructs an inverted feature for the specifed value which occurs at a single index.
     * @param value The value.
     * @param index The index where the value occurs.
     */
    public InvertedFeature(double value, int index) {
        this.value = value;
        this.index = index;
    }

    private InvertedFeature(InvertedFeature other) {
        this.value = other.value;
        this.curSize = other.curSize;
        this.index = other.index;
        if (other.indices != null) {
            this.indices = Arrays.copyOf(other.indices,other.indices.length);
        } else {
            this.indices = null;
        }
    }

    /**
     * Adds an index where the feature value occurs.
     * @param index The index.
     */
    public void add(int index) {
        if (indices == null) {
            initArrays();
        }
        append(index);
    }

    private void append(int index) {
        if (curSize == indices.length) {
            int newSize = indices.length + (indices.length >> 1);
            indices = Arrays.copyOf(indices,newSize);
        }
        indices[curSize] = index;
        curSize++;
    }

    /**
     * Gets the indices where this feature value occurs.
     * @return The indices.
     */
    public int[] indices() {
        if (indices != null) {
            return indices;
        } else {
            int[] ret = new int[1];
            ret[0] = index;
            return ret;
        }
    }

    /**
     * Fixes the size of the backing array.
     * <p>
     * Used when all the feature values have been observed.
     */
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
                left = new InvertedFeature(value,leftIndices[0]);
            } else {
                left = new InvertedFeature(value, Arrays.copyOf(leftIndices, leftSize));
            }
            if (rightSize == 0) {
                right = null;
            } else if (rightSize == 1) {
                right = new InvertedFeature(value,rightIndices[0]);
            } else {
                right = new InvertedFeature(value, Arrays.copyOf(rightIndices, rightSize));
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
                return new Pair<>(new InvertedFeature(value,index),null);
            } else {
                allLeftIndices.array = bufferArray;
                allLeftIndices.size = 0;
                buffer.array = allLeftArray;
                buffer.size = allLeftSize;
                return new Pair<>(null,new InvertedFeature(value,index));
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
    public String toString() {
        if (indices != null) {
            return "InvertedFeature(value=" + value + ",size=" + curSize + ",indices=" + Arrays.toString(indices) + ")";
        } else {
            return "InvertedFeature(value=" + value + ",size=" + curSize + ",index=" + index + ")";
        }
    }

    /**
     * Copies this inverted feature.
     * @return A copy of this feature.
     */
    public InvertedFeature deepCopy() {
        return new InvertedFeature(this);
    }
}
