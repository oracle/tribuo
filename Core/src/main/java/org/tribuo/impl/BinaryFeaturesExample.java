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

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.Output;
import org.tribuo.VariableInfo;
import org.tribuo.transform.TransformerMap;
import org.tribuo.util.Merger;

import com.oracle.labs.mlrg.olcut.util.SortUtil;

/**
 * An {@link Example} backed by a single array of feature names. This
 * implementation is modeled after {@link ArrayExample} but does not store
 * feature values because it assumes only binary features - i.e. features with a
 * feature value of 1.0. The following methods throw an
 * {@link UnsupportedOperationException}:
 * <ul>
 * <li>{@link #densify(List)}</li>
 * <li>{@link #densify(FeatureMap)}</li>
 * <li>{@link #set(Feature)}</li>
 * <li>{@link #transform(TransformerMap)}</li>
 * </ul>
 *
 * @param <T>
 */
public final class BinaryFeaturesExample<T extends Output<T>> extends Example<T> {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(BinaryFeaturesExample.class.getName());

    public static final int DEFAULT_SIZE = 10;

    protected String[] featureNames;

    protected int size = 0;

    /**
     * Constructs an example from an output and a weight, with an initial size for
     * the feature arrays.
     * 
     * @param output      The output.
     * @param weight      The weight.
     * @param initialSize The initial size of the feature arrays.
     */
    public BinaryFeaturesExample(T output, float weight, int initialSize) {
        super(output,weight);
        featureNames = new String[initialSize];
    }

    /**
     * Constructs an example from an output, a weight and the metadata.
     * 
     * @param output   The output.
     * @param weight   The weight.
     * @param metadata The metadata.
     */
    public BinaryFeaturesExample(T output, float weight, Map<String,Object> metadata) {
        super(output,weight,metadata);
        featureNames = new String[DEFAULT_SIZE];
    }

    /**
     * Constructs an example from an output and a weight.
     * 
     * @param output The output.
     * @param weight The example weight.
     */
    public BinaryFeaturesExample(T output, float weight) {
        super(output,weight);
        featureNames = new String[DEFAULT_SIZE];
    }

    /**
     * Constructs an example from an output and the metadata.
     * 
     * @param output   The output.
     * @param metadata The metadata.
     */
    public BinaryFeaturesExample(T output, Map<String,Object> metadata) {
        super(output,metadata);
        featureNames = new String[DEFAULT_SIZE];
    }

    /**
     * Constructs an example from an output.
     * 
     * @param output The output.
     */
    public BinaryFeaturesExample(T output) {
        super(output);
        featureNames = new String[DEFAULT_SIZE];
    }

    /**
     * Constructs an example from an output and an array of names. This is currently
     * the most efficient constructor.
     * 
     * @param output The output.
     * @param names  The feature names.
     */
    public BinaryFeaturesExample(T output, String[] names) {
        super(output);
        size = names.length;
        featureNames = Arrays.copyOf(names,names.length);
        sort();
    }

    /**
     * Constructs an example from an output and a list of features. This constructor
     * will throw an {@link IllegalArgumentException} if any of the features have a
     * feature value that does not equal 1.0.
     * 
     * @param output   The output (e.g. label) of the example
     * @param features The list of features.
     */
    public BinaryFeaturesExample(T output, List<? extends Feature> features) {
        super(output);
        size = features.size();
        featureNames = new String[size];

        int i = 0;
        for (Feature f : features) {
            checkIsBinary(f);
            featureNames[i] = f.getName();
            i++;
        }

        sort();
    }

    /**
     * Copy constructor. This constructor will throw an
     * {@link IllegalArgumentException} if any of the features have a feature value
     * that does not equal 1.0.
     * 
     * @param other The example to copy.
     */
    public BinaryFeaturesExample(Example<T> other) {
        super(other);
        featureNames = new String[other.size()];
        for (Feature f : other) {
            checkIsBinary(f);
            featureNames[size] = f.getName();
            size++;
        }
    }

    /**
     * Clones an example's features, but uses the supplied output and weight.
     * @param output The output to use.
     * @param other The features to use.
     * @param weight The weight to use.
     * @param <U> The output type of the other example.
     */
    public <U extends Output<U>> BinaryFeaturesExample(T output, Example<U> other, float weight) {
        super(output,weight);
        featureNames = new String[other.size()];
        for (Feature f : other) {
            checkIsBinary(f);
            featureNames[size] = f.getName();
            size++;
        }
    }

    /**
     * Adds a single feature.
     * @param name The name of the feature.
     * @param value The value of the feature.
     */
    public void add(String name) {
        if (size >= featureNames.length) {
            growArray();
        }
        //
        // TODO: find the right insertion position, System.arraycopy
        // everything up one and then write the new value.
        featureNames[size] = name;
        size++;
        sort();
    }

    public static boolean isBinary(Feature feature) {
        return feature.getValue() == 1.0;
    }

    public static void checkIsBinary(Feature feature) {
        if(!isBinary(feature)) {
            throw new IllegalArgumentException("non-binary features are not allowed in BinaryFeaturesExample: value="+feature.getValue());
        }
    }

    /**
     * Adds a feature to this example. This method will throw an {@link IllegalArgumentException} if
     * any of the features have a feature value that does not equal 1.0.
     * 
     * @param feature The feature to add to this example.
     */
    @Override
    public void add(Feature feature) {
        checkIsBinary(feature);
        add(feature.getName());
    }

    /**
     * Adds a collection of features to this example. This method will throw an
     * {@link IllegalArgumentException} if any of the features have a feature value
     * that does not equal 1.0.
     * 
     * @param features The features to add to this example.
     */
    @Override
    public void addAll(Collection<? extends Feature> features) {
        if (size + features.size() >= featureNames.length) {
            growArray(size+features.size());
        }
        for (Feature f : features) {
            checkIsBinary(f);
            featureNames[size] = f.getName();
            size++;
        }
        sort();
    }

    /**
     * Grows the backing arrays storing the names.
     * @param minCapacity The new minimum capacity required.
     */
    protected void growArray(int minCapacity) {
        int newCapacity = newCapacity(minCapacity);
        featureNames = Arrays.copyOf(featureNames,newCapacity);
    }

    /**
     * Grows the backing arrays by size+1.
     */
    protected void growArray() {
        growArray(size+1);
    }

    /**
     * Returns a capacity at least as large as the given minimum capacity.
     * Returns the current capacity increased by 50% if that suffices.
     * Will not return a capacity greater than MAX_ARRAY_SIZE unless
     * the given minimum capacity is greater than MAX_ARRAY_SIZE.
     *
     * @param minCapacity the desired minimum capacity
     * @throws OutOfMemoryError if minCapacity is less than zero
     * @return The new capacity.
     */
    protected int newCapacity(int minCapacity) {
        // overflow-conscious code
        int oldCapacity = featureNames.length;
        int newCapacity = oldCapacity + (oldCapacity >> 1);
        if (newCapacity - minCapacity <= 0) {
            if (minCapacity < 0) {
                // overflow
                throw new OutOfMemoryError();
            }
            return minCapacity;
        }
        return newCapacity;
    }

    /**
     * Sorts the feature list to maintain the lexicographic order invariant.
     */
    @Override
    protected void sort() {
        int[] sortedIndices = SortUtil.argsort(featureNames,0,size,true);

        String[] newNames = Arrays.copyOf(featureNames,size);
        for (int i = 0; i < sortedIndices.length; i++) {
            featureNames[i] = newNames[sortedIndices[i]];
        }
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void removeFeatures(List<Feature> featureList) {
        Map<String,Integer> map = new HashMap<>();
        for (int i = 0; i < featureNames.length; i++) {
            map.put(featureNames[i],i);
        }

        PriorityQueue<Integer> removeQueue = new PriorityQueue<>();
        for (Feature f : featureList) {
            Integer i = map.get(f.getName());
            if (i != null) {
                removeQueue.add(i);
            }
        }

        String[] newNames = new String[size-removeQueue.size()];

        int source = 0;
        int dest = 0;
        while (!removeQueue.isEmpty()) {
            int curRemoveIdx = removeQueue.poll();
            while (source < curRemoveIdx) {
                newNames[dest] = featureNames[source];
                source++;
                dest++;
            }
            source++;
        }
        while (source < size) {
            newNames[dest] = featureNames[source];
            source++;
            dest++;
        }
        featureNames = newNames;
        size = featureNames.length;
    }

    @Override
    public boolean validateExample() {
        if (size == 0) {
            return false;
        } else {
            Set<String> names = new HashSet<>(Arrays.asList(featureNames).subList(0, size));
            return names.size() == size;
        }
    }

    @Override
    public BinaryFeaturesExample<T> copy() {
        return new BinaryFeaturesExample<>(this);
    }

    @Override
    public Feature lookup(String i) {
        int index = Arrays.binarySearch(featureNames,0,size,i);
        if (index < 0) {
            return null;
        } else {
            return new Feature(featureNames[index], 1.0);
        }
    }

    @Override
    public void set(Feature feature) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void transform(TransformerMap transformerMap) {
        throw new UnsupportedOperationException();
    }

    /**
     * @param merger this parameter is ignored because we do not store feature values in this class.  So, the 'merging' can be thought of as a logical AND.
     */
    @Override
    public void reduceByName(Merger merger) {
        if (size > 0) {
            List<String> newNames = Arrays.stream(featureNames, 0, size).distinct().collect(Collectors.toList());
            Collections.sort(newNames);
            featureNames = newNames.toArray(new String[newNames.size()]);
            size = featureNames.length;
        } else {
            logger.finer("Reducing an example with no features.");
        }
    }

    @Override
    public void densify(List<String> featureList) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void densify(FeatureMap fMap) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Iterator<Feature> iterator() {
        return new BinaryFeaturesExampleIterator();
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        builder.append("ArrayExample(numFeatures=");
        builder.append(size);
        builder.append(",output=");
        builder.append(output);
        builder.append(",weight=");
        builder.append(weight);
        if (metadata != null) {
            builder.append(",metadata=");
            builder.append(metadata.toString());
        }
        builder.append(",features=[");
        for(int i = 0; i < size; i++) {
            builder.append('(').append(featureNames[i]).append(", ").append(1.0).append(')');
            if(i != 0) {
                builder.append(", ");
            }
        }
        //builder.append(features.toString());
        builder.append("])");

        return builder.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BinaryFeaturesExample)) return false;
        BinaryFeaturesExample<?> that = (BinaryFeaturesExample<?>) o;
        if (Objects.equals(metadata,that.metadata) && output.getClass().equals(that.output.getClass())) {
            @SuppressWarnings("unchecked") //guarded by a getClass.
            boolean outputTest = output.fullEquals((T)that.output);
            if(outputTest && size == that.size) {
                //we do not use Arrays.equals here because these are "backing arrays" which could be different sizes 
                for(int i=0; i<size; i++) {
                    if(!this.featureNames[i].equals(that.featureNames[i])) return false;
                }
                return true;
            }
            return false;
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(size);
        result = 31 * result + output.hashCode();
        //we don't use Arrays.hashCode here because featureNames 
        //is a backing array and its length could be arbitrarily diverging 
        //from the member size.  
        for(int i=0; i<size; i++) {
            result = 31 * result + featureNames[i].hashCode();
        }
        return result;
    }

    class BinaryFeaturesExampleIterator implements Iterator<Feature> {
        int pos = 0;

        @Override
        public boolean hasNext() {
            return pos < size;
        }

        @Override
        public Feature next() {
            Feature f = new Feature(featureNames[pos], 1.0);
            pos++;
            return f;
        }
    }

    @Override
    public void canonicalize(FeatureMap featureMap) {
        for(int i=0; i< featureNames.length; i++) {
            VariableInfo vi = featureMap.get(featureNames[i]);
            if(vi != null) {
                featureNames[i] = vi.getName();
            }
        }
    }
}
