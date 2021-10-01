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

package org.tribuo.impl;

import com.oracle.labs.mlrg.olcut.util.SortUtil;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.util.Merger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.PriorityQueue;

/**
 * A version of ArrayExample which also has the id numbers.
 * <p>
 * Used in feature selection to provide log n lookups. May be used
 * elsewhere in the future as a performance optimisation.
 * <p>
 * Note: output id caching is only valid with single dimensional {@link Output}s like ClusterID, Event and Label.
 * Other outputs may return -1 from {@link #getOutputID()}.
 */
public class IndexedArrayExample<T extends Output<T>> extends ArrayExample<T> {
    private static final long serialVersionUID = 1L;

    protected int[] featureIDs;

    protected final int outputID;

    private final ImmutableFeatureMap featureMap;

    private final ImmutableOutputInfo<T> outputMap;

    /**
     * Copy constructor.
     * @param other The example to copy.
     */
    public IndexedArrayExample(IndexedArrayExample<T> other) {
        super(other.getOutput(),other.getWeight(),other.getMetadata());
        featureNames = Arrays.copyOf(other.featureNames,other.featureNames.length);
        featureIDs = Arrays.copyOf(other.featureIDs,other.size());
        featureValues = Arrays.copyOf(other.featureValues,other.featureValues.length);
        featureMap = other.featureMap;
        outputMap = other.outputMap;
        outputID = outputMap.getID(output);
        size = other.size;
    }

    /**
     * This constructor removes unknown features.
     * @param other The example to copy from.
     * @param featureMap The feature map.
     * @param outputMap The output info.
     */
    public IndexedArrayExample(Example<T> other, ImmutableFeatureMap featureMap, ImmutableOutputInfo<T> outputMap) {
        super(other);
        this.featureIDs = new int[other.size()];
        this.featureMap = featureMap;
        this.outputMap = outputMap;
        this.outputID = outputMap.getID(output);
        for (int i = 0; i < featureNames.length; i++) {
            featureIDs[i] = featureMap.getID(featureNames[i]);
        }
        int[] newIDs = new int[other.size()];
        String[] newNames = new String[other.size()];
        double[] newValues = new double[other.size()];
        int counter = 0;
        for (int i = 0; i < featureIDs.length; i++) {
            if (featureIDs[i] != -1) {
                newIDs[counter] = featureIDs[i];
                newValues[counter] = featureValues[i];
                newNames[counter] = featureNames[i];
                counter++;
            }
        }
        size = counter;
        featureNames = newNames;
        featureIDs = newIDs;
        featureValues = newValues;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof IndexedArrayExample)) return false;
        if (!super.equals(o)) return false;
        IndexedArrayExample<?> that = (IndexedArrayExample<?>) o;
        return outputID == that.outputID &&
                Arrays.equals(featureIDs, that.featureIDs) &&
                featureMap.equals(that.featureMap) &&
                outputMap.equals(that.outputMap);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(super.hashCode(), outputID, featureMap, outputMap);
        result = 31 * result + Arrays.hashCode(featureIDs);
        return result;
    }

    @Override
    protected void growArray(int minCapacity) {
        int newCapacity = newCapacity(minCapacity);
        featureNames = Arrays.copyOf(featureNames,newCapacity);
        featureIDs = Arrays.copyOf(featureIDs,newCapacity);
        featureValues = Arrays.copyOf(featureValues,newCapacity);
    }

    @Override
    public void add(Feature feature) {
        if (size >= featureNames.length) {
            growArray();
        }
        featureNames[size] = feature.getName();
        featureIDs[size] = featureMap.getID(feature.getName());
        featureValues[size] = feature.getValue();
        size++;
        sort();
    }

    @Override
    public void addAll(Collection<? extends Feature> features) {
        if (size + features.size() >= featureNames.length) {
            growArray(size+features.size());
        }
        for (Feature f : features) {
            featureNames[size] = f.getName();
            featureIDs[size] = featureMap.getID(f.getName());
            featureValues[size] = f.getValue();
            size++;
        }
        sort();
    }

    @Override
    protected void sort() {
        int[] sortedIndices = SortUtil.argsort(featureNames,0,size,true);

        String[] newNames = Arrays.copyOf(featureNames,size);
        int[] newIDs = Arrays.copyOf(featureIDs,size);
        double[] newValues = Arrays.copyOf(featureValues,size);
        for (int i = 0; i < sortedIndices.length; i++) {
            featureNames[i] = newNames[sortedIndices[i]];
            featureIDs[i] = newIDs[sortedIndices[i]];
            featureValues[i] = newValues[sortedIndices[i]];
        }
    }

    @Override
    public void reduceByName(Merger merger) {
        if (size > 0) {
            int[] sortedIndices = SortUtil.argsort(featureNames, 0, size, true);
            String[] newNames = new String[featureNames.length];
            int[] newIDs = new int[featureIDs.length];
            double[] newValues = new double[featureNames.length];
            for (int i = 0; i < sortedIndices.length; i++) {
                newNames[i] = featureNames[sortedIndices[i]];
                newIDs[i] = featureIDs[sortedIndices[i]];
                newValues[i] = featureValues[sortedIndices[i]];
            }
            featureNames[0] = newNames[0];
            featureIDs[0] = newIDs[0];
            featureValues[0] = newValues[0];
            int dest = 0;
            for (int i = 1; i < size; i++) {
                while ((i < size) && newNames[i].equals(featureNames[dest])) {
                    featureValues[dest] = merger.merge(featureValues[dest], newValues[i]);
                    i++;
                }
                if (i < size) {
                    dest++;
                    featureNames[dest] = newNames[i];
                    featureIDs[dest] = newIDs[i];
                    featureValues[dest] = newValues[i];
                }
            }
            size = dest + 1;
        }
    }

    @Override
    public void removeFeatures(List<Feature> featureList) {
        Map<String,List<Integer>> map = new HashMap<>();
        for (int i = 0; i < featureNames.length; i++) {
            List<Integer> list = map.computeIfAbsent(featureNames[i],(k) -> new ArrayList<>());
            list.add(i);
        }

        PriorityQueue<Integer> removeQueue = new PriorityQueue<>();
        for (Feature f : featureList) {
            List<Integer> i = map.get(f.getName());
            if (i != null) {
                // If we've found this feature remove it from the map to prevent double counting
                map.remove(f.getName());
                removeQueue.addAll(i);
            }
        }

        String[] newNames = new String[size-removeQueue.size()];
        int[] newIDs = new int[size-removeQueue.size()];
        double[] newValues = new double[size-removeQueue.size()];

        int source = 0;
        int dest = 0;
        while (!removeQueue.isEmpty()) {
            int curRemoveIdx = removeQueue.poll();
            while (source < curRemoveIdx) {
                newNames[dest] = featureNames[source];
                newIDs[dest] = featureIDs[source];
                newValues[dest] = featureValues[source];
                source++;
                dest++;
            }
            source++;
        }
        while (source < size) {
            newNames[dest] = featureNames[source];
            newIDs[dest] = featureIDs[source];
            newValues[dest] = featureValues[source];
            source++;
            dest++;
        }
        featureNames = newNames;
        featureIDs = newIDs;
        featureValues = newValues;
        size = featureNames.length;
    }

    /**
     * Does this example contain a feature with id i.
     * @param i The index to check.
     * @return True if the example contains the id.
     */
    public boolean contains(int i) {
        return Arrays.binarySearch(featureIDs,i) > -1;
    }

    @Override
    public IndexedArrayExample<T> copy() {
        return new IndexedArrayExample<>(this);
    }

    @Override
    public void densify(List<String> featureList) {
        if (featureList.size() != featureMap.size()) {
            throw new IllegalArgumentException("Densifying an example with a different feature map");
        }
        // Ensure we have enough space.
        if (featureList.size() > featureNames.length) {
            growArray(featureList.size());
        }
        int insertedCount = 0;
        int curPos = 0;
        for (String curName : featureList) {
            // If we've reached the end of our old feature set, just insert.
            if (curPos == size) {
                featureNames[size + insertedCount] = curName;
                featureIDs[size + insertedCount] = featureMap.getID(curName);
                insertedCount++;
            } else {
                // Check to see if our insertion candidate is the same as the current feature name.
                int comparison = curName.compareTo(featureNames[curPos]);
                if (comparison < 0) {
                    // If it's earlier, insert it.
                    featureNames[size + insertedCount] = curName;
                    featureIDs[size + insertedCount] = featureMap.getID(curName);
                    insertedCount++;
                } else if (comparison == 0) {
                    // Otherwise just bump our pointer, we've already got this feature.
                    curPos++;
                }
            }
        }
        // Bump the size up by the number of inserted features.
        size += insertedCount;
        // Sort the features
        sort();
    }

    /**
     * Gets the feature at internal index i.
     * @param i The internal index.
     * @return The feature index.
     */
    public int getIdx(int i) {
        return featureIDs[i];
    }

    /**
     * Gets the output id dimension number.
     * @return The output id.
     */
    public int getOutputID() {
        return outputID;
    }

    /**
     * Iterator over the feature ids and values.
     * @return The feature ids and values.
     */
    public Iterable<FeatureTuple> idIterator() {
        return ArrayIndexedExampleIterator::new;
    }

    /**
     * A tuple of the feature name, id and value.
     */
    public static class FeatureTuple {
        public String name;
        public int id;
        public double value;

        public FeatureTuple() { }

        public FeatureTuple(String name, int id, double value) {
            this.name = name;
            this.id = id;
            this.value = value;
        }
    }

    class ArrayIndexedExampleIterator implements Iterator<FeatureTuple> {
        int pos = 0;
        FeatureTuple tuple = new FeatureTuple();

        @Override
        public boolean hasNext() {
            return pos < size;
        }

        @Override
        public FeatureTuple next() {
            if (!hasNext()) {
                throw new NoSuchElementException("Off the end of the iterator.");
            }
            tuple.name = featureNames[pos];
            tuple.id = featureIDs[pos];
            tuple.value = featureValues[pos];
            pos++;
            return tuple;
        }
    }
}
