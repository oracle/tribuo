/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.SortUtil;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.Output;
import org.tribuo.VariableInfo;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.ExampleDataProto;
import org.tribuo.protos.core.ExampleProto;
import org.tribuo.transform.Transformer;
import org.tribuo.transform.TransformerMap;
import org.tribuo.util.Merger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.logging.Logger;

/**
 * An {@link Example} backed by two arrays, one of String and one of double.
 */
public class ArrayExample<T extends Output<T>> extends Example<T> {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(ArrayExample.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Default initial size of the backing arrays.
     */
    public static final int DEFAULT_SIZE = 10;

    /**
     * Feature names array.
     */
    protected String[] featureNames;

    /**
     * Feature values array.
     */
    protected double[] featureValues;

    /**
     * Number of valid features in this example.
     */
    protected int size = 0;

    /**
     * Constructs an example from an output and a weight, with an initial
     * size for the feature arrays.
     * @param output The output.
     * @param weight The weight.
     * @param initialSize The initial size of the feature arrays.
     */
    public ArrayExample(T output, float weight, int initialSize) {
        super(output,weight);
        featureNames = new String[initialSize];
        featureValues = new double[initialSize];
    }

    /**
     * Constructs an example from an output, a weight and the metadata.
     * @param output The output.
     * @param weight The weight.
     * @param metadata The metadata.
     */
    public ArrayExample(T output, float weight, Map<String,Object> metadata) {
        super(output,weight,metadata);
        featureNames = new String[DEFAULT_SIZE];
        featureValues = new double[DEFAULT_SIZE];
    }

    /**
     * Constructs an example from an output and a weight.
     * @param output The output.
     * @param weight The example weight.
     */
    public ArrayExample(T output, float weight) {
        super(output,weight);
        featureNames = new String[DEFAULT_SIZE];
        featureValues = new double[DEFAULT_SIZE];
    }

    /**
     * Constructs an example from an output and the metadata.
     * @param output The output.
     * @param metadata The metadata.
     */
    public ArrayExample(T output, Map<String,Object> metadata) {
        super(output,metadata);
        featureNames = new String[DEFAULT_SIZE];
        featureValues = new double[DEFAULT_SIZE];
    }

    /**
     * Constructs an example from an output.
     * @param output The output.
     */
    public ArrayExample(T output) {
        super(output);
        featureNames = new String[DEFAULT_SIZE];
        featureValues = new double[DEFAULT_SIZE];
    }

    /**
     * Constructs an example from an output, an array of names and an array of values.
     * This is currently the most efficient constructor.
     * @param output The output.
     * @param names The feature names.
     * @param values The feature values.
     */
    public ArrayExample(T output, String[] names, double[] values) {
        super(output);
        if (names.length != values.length) {
            throw new IllegalArgumentException("names.length != values.length, names = " + names.length + ", values = " + values.length);
        }

        size = names.length;
        featureNames = Arrays.copyOf(names,names.length);
        featureValues = Arrays.copyOf(values,values.length);

        sort();
    }

    /**
     * Constructs an example from an output and a list of features.
     * @param output The output.
     * @param features The list of features.
     */
    public ArrayExample(T output, List<? extends Feature> features) {
        super(output);
        size = features.size();
        featureNames = new String[size];
        featureValues = new double[size];

        int i = 0;
        for (Feature f : features) {
            featureNames[i] = f.getName();
            featureValues[i] = f.getValue();
            i++;
        }

        sort();
    }

    /**
     * Copy constructor.
     * @param other The example to copy.
     */
    public ArrayExample(Example<T> other) {
        super(other);
        if(other instanceof ArrayExample) {
            ArrayExample< T> otherArr = (ArrayExample<T>) other;
            featureNames = Arrays.copyOf(otherArr.featureNames, otherArr.size);
            featureValues = Arrays.copyOf(otherArr.featureValues, otherArr.size);
            size = otherArr.size;
        } else {
            featureNames = new String[other.size()];
            featureValues = new double[other.size()];
            for(Feature f : other) {
                featureNames[size] = f.getName();
                featureValues[size] = f.getValue();
                size++;
            }
        }
    }

    /**
     * Clones an example's features, but uses the supplied output and weight.
     * @param output The output to use.
     * @param other The features to use.
     * @param weight The weight to use.
     * @param <U> The output type of the other example.
     */
    public <U extends Output<U>> ArrayExample(T output, Example<U> other, float weight) {
        super(output,weight);
        featureNames = new String[other.size()];
        featureValues = new double[other.size()];
        for (Feature f : other) {
            featureNames[size] = f.getName();
            featureValues[size] = f.getValue();
            size++;
        }
    }

    /**
     * Deserialization constructor.
     * @param output The output.
     * @param weight The weight.
     * @param featureNames The feature names.
     * @param featureValues The feature values.
     * @param metadata The metadata map.
     */
    private ArrayExample(T output, float weight, String[] featureNames, double[] featureValues, Map<String, String> metadata) {
        super(output,weight);
        this.featureNames = Arrays.copyOf(featureNames,featureNames.length);
        this.featureValues = Arrays.copyOf(featureValues,featureValues.length);
        this.size = featureNames.length;
        for (Map.Entry<String, String> e : metadata.entrySet()) {
            setMetadataValue(e.getKey(), e.getValue());
        }
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    @SuppressWarnings({"rawtypes","unchecked"})
    public static ArrayExample<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        ExampleDataProto proto = message.unpack(ExampleDataProto.class);
        if (proto.getFeatureNameCount() != proto.getFeatureValueCount()) {
            throw new IllegalStateException("Invalid protobuf, different numbers of feature names and values, found " + proto.getFeatureNameCount() + " names and " + proto.getFeatureValueCount() + " values.");
        }
        Output<?> output = ProtoUtil.deserialize(proto.getOutput());
        String[] featureNames = new String[proto.getFeatureNameCount()];
        double[] featureValues = new double[proto.getFeatureValueCount()];
        for (int i = 0; i < proto.getFeatureNameCount(); i++) {
            featureNames[i] = proto.getFeatureName(i);
            featureValues[i] = proto.getFeatureValue(i);
        }
        return new ArrayExample(output,proto.getWeight(),featureNames,featureValues,proto.getMetadataMap());
    }

    /**
     * Adds a single feature.
     * @param name The name of the feature.
     * @param value The value of the feature.
     */
    public void add(String name, double value) {
        if (size >= featureNames.length) {
            growArray();
        }
        //
        // TODO: find the right insertion position, System.arraycopy
        // everything up one and then write the new value.
        featureNames[size] = name;
        featureValues[size] = value;
        size++;
        sort();
    }

    @Override
    public void add(Feature feature) {
        add(feature.getName(),feature.getValue());
    }

    @Override
    public void addAll(Collection<? extends Feature> features) {
        if (size + features.size() >= featureNames.length) {
            growArray(size+features.size());
        }
        for (Feature f : features) {
            featureNames[size] = f.getName();
            featureValues[size] = f.getValue();
            size++;
        }
        sort();
    }

    /**
     * Grows the backing arrays storing the names and values.
     * @param minCapacity The new minimum capacity required.
     */
    protected void growArray(int minCapacity) {
        int newCapacity = newCapacity(minCapacity);
        featureNames = Arrays.copyOf(featureNames,newCapacity);
        featureValues = Arrays.copyOf(featureValues,newCapacity);
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
        double[] newValues = Arrays.copyOf(featureValues,size);
        for (int i = 0; i < sortedIndices.length; i++) {
            featureNames[i] = newNames[sortedIndices[i]];
            featureValues[i] = newValues[sortedIndices[i]];
        }
    }

    /**
     * Returns a copy of the feature values array at the specific size.
     * @param newSize The new size.
     * @return A copy of the feature values.
     */
    public double[] copyValues(int newSize) {
        return Arrays.copyOf(featureValues,newSize);
    }

    @Override
    public int size() {
        return size;
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
        double[] newValues = new double[size-removeQueue.size()];

        int source = 0;
        int dest = 0;
        while (!removeQueue.isEmpty()) {
            int curRemoveIdx = removeQueue.poll();
            while (source < curRemoveIdx) {
                newNames[dest] = featureNames[source];
                newValues[dest] = featureValues[source];
                source++;
                dest++;
            }
            source++;
        }
        while (source < size) {
            newNames[dest] = featureNames[source];
            newValues[dest] = featureValues[source];
            source++;
            dest++;
        }
        featureNames = newNames;
        featureValues = newValues;
        size = featureNames.length;
    }

    @Override
    public void reduceByName(Merger merger) {
        if (size > 0) {
            int[] sortedIndices = SortUtil.argsort(featureNames,0,size,true);
            String[] newNames = new String[featureNames.length];
            double[] newValues = new double[featureNames.length];
            for (int i = 0; i < sortedIndices.length; i++) {
                newNames[i] = featureNames[sortedIndices[i]];
                newValues[i] = featureValues[sortedIndices[i]];
            }
            featureNames[0] = newNames[0];
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
                    featureValues[dest] = newValues[i];
                }
            }
            size = dest + 1;
        } else {
            logger.finer("Reducing an example with no features.");
        }
    }

    @Override
    public boolean validateExample() {
        if (size == 0) {
            return false;
        } else {
            for (int i = 0; i < size; i++) {
                if (Double.isNaN(featureValues[i])) {
                    return false;
                }
            }
            Set<String> names = new HashSet<>(Arrays.asList(featureNames).subList(0, size));
            return names.size() == size;
        }
    }

    @Override
    public ArrayExample<T> copy() {
        return new ArrayExample<>(this);
    }

    @Override
    public Feature lookup(String i) {
        int index = Arrays.binarySearch(featureNames,0,size,i);
        if (index < 0) {
            return null;
        } else {
            return new Feature(featureNames[index],featureValues[index]);
        }
    }

    @Override
    public void set(Feature feature) {
        int index = Arrays.binarySearch(featureNames,0,size,feature.getName());
        if (index < 0) {
            throw new IllegalArgumentException("Feature " + feature + " not found in example.");
        } else {
            featureValues[index] = feature.getValue();
        }
    }

    @Override
    public void transform(TransformerMap transformerMap) {
        if(transformerMap.size() < size) {
            //
            // We have fewer transformers than feature names, so let's
            // iterate through the map and find the features.
            for(Map.Entry<String, List<Transformer>> e : transformerMap.entrySet()) {
                int index = Arrays.binarySearch(featureNames, 0, size, e.getKey());
                if(index >= 0) {
                    double value = featureValues[index];
                    for(Transformer t : e.getValue()) {
                        value = t.transform(value);
                    }
                    featureValues[index] = value;
                }
            }
        } else {
            //
            // We have more transformers, so let's fetch them by name.
            for(int i = 0; i < size; i++) {
                List<Transformer> l = transformerMap.get(featureNames[i]);
                if(l != null) {
                    double value = featureValues[i];
                    for(Transformer t : l) {
                        value = t.transform(value);
                    }
                    featureValues[i] = value;
                }
            }
        }
    }

    @Override
    public boolean isDense(FeatureMap fMap) {
        if (fMap.size() == size()) {
            // We've got the right number of features
            for (String name : featureNames) {
                if (fMap.get(name) == null) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    @Override
    public void densify(List<String> featureList) {
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
                insertedCount++;
            } else {
                // Check to see if our insertion candidate is the same as the current feature name.
                int comparison = curName.compareTo(featureNames[curPos]);
                if (comparison < 0) {
                    // If it's earlier, insert it.
                    featureNames[size + insertedCount] = curName;
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

    @Override
    public Iterator<Feature> iterator() {
        return new ArrayExampleIterator();
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
            builder.append('(').append(featureNames[i]).append(", ").append(featureValues[i]).append(')');
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
        if (!(o instanceof ArrayExample)) return false;
        ArrayExample<?> that = (ArrayExample<?>) o;
        if (Objects.equals(metadata,that.metadata) && output.getClass().equals(that.output.getClass())) {
            @SuppressWarnings("unchecked") //guarded by a getClass.
            boolean outputTest = output.fullEquals((T)that.output);
            if(outputTest && size == that.size) {
                //we do not use Arrays.equals here because these are "backing arrays" which could be different sizes 
                for(int i=0; i<size; i++) {
                    if(!this.featureNames[i].equals(that.featureNames[i])) return false;
                    if(this.featureValues[i] != that.featureValues[i]) return false;
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
        //we don't use Arrays.hashCode here because featureNames and featureValues
        //are backing arrays and the length of each could be arbitrarily diverging 
        //from the member size.  
        for(int i=0; i<size; i++) {
            result = 31 * result + featureNames[i].hashCode();
            result = 31 * result + Double.hashCode(featureValues[i]);
        }
        return result;
    }

    class ArrayExampleIterator implements Iterator<Feature> {
        int pos = 0;

        @Override
        public boolean hasNext() {
            return pos < size;
        }

        @Override
        public Feature next() {
            if (!hasNext()) {
                throw new NoSuchElementException("Iterator exhausted at position " + pos);
            }
            Feature f = new Feature(featureNames[pos],featureValues[pos]);
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

    @Override
    public ExampleProto serialize() {
        ExampleProto.Builder builder = ExampleProto.newBuilder();

        builder.setClassName(ArrayExample.class.getName());
        builder.setVersion(CURRENT_VERSION);
        ExampleDataProto.Builder exampleBuilder = ExampleDataProto.newBuilder();
        exampleBuilder.setWeight(weight);
        exampleBuilder.setOutput(output.serialize());
        for (int i = 0; i < size; i++) {
            exampleBuilder.addFeatureName(featureNames[i]);
            exampleBuilder.addFeatureValue(featureValues[i]);
        }
        if (metadata != null) {
            for (Map.Entry<String, Object> e : metadata.entrySet()) {
                if (!(e.getValue() instanceof String)) {
                    logger.warning("Serializing non-string metadata for key '" + e.getKey() + "' of type " + e.getValue().getClass());
                }
                exampleBuilder.putMetadata(e.getKey(), e.getValue().toString());
            }
        }

        builder.setSerializedData(Any.pack(exampleBuilder.build()));

        return builder.build();
    }
}
