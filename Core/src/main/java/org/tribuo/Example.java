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

package org.tribuo;

import org.tribuo.hash.HashedFeatureMap;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.ExampleProto;
import org.tribuo.transform.Transformer;
import org.tribuo.transform.TransformerMap;
import org.tribuo.util.Merger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * An example used for training and evaluation. Examples have a true output
 * associated with them or an instance from {@link OutputFactory#getUnknownOutput()}
 * and a list of features that can be used for prediction.
 * <p>
 * An example is a sorted set of features, sorted by the String comparator on the feature
 * name.
 * <p>
 * Examples have metadata associated with them, stored as a map from a String key, to
 * an Object value. This metadata is append only for any given example, and the metadata
 * values should be immutable (as they will be referenced rather than copied when an
 * example is copied). Values with are not {@link Serializable} will cause exceptions if
 * the example is serialized. Note protobuf serialization only supports string values, and will
 * coerce all values to strings before serialization. In a future release metadata will only
 * support {@link String} values.
 * @param <T> The type of output that this example contains.
 */
public abstract class Example<T extends Output<T>> implements Iterable<Feature>, ProtoSerializable<ExampleProto>, Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * The default initial size of the metadata map.
     */
    protected static final int DEFAULT_METADATA_SIZE = 2;

    /**
     * The default weight.
     */
    public static final float DEFAULT_WEIGHT = 1.0f;

    /**
     * By convention the example name is stored using this metadata key.
     * <p>
     * Note: not all examples are named.
     */
    public static final String NAME = "name";

    /**
     * The output associated with this example.
     */
    protected final T output;

    /**
     * The weight associated with this example.
     */
    protected float weight = DEFAULT_WEIGHT;

    /**
     * The example metadata.
     */
    protected Map<String,Object> metadata = null;

    /**
     * Construct an empty example using the supplied output, weight and metadata.
     * @param output The output.
     * @param weight The weight.
     * @param metadata The metadata.
     */
    protected Example(T output, float weight, Map<String,Object> metadata) {
        this.output = output;
        this.weight = weight;
        if (metadata != null && !metadata.isEmpty()) {
            this.metadata = new HashMap<>(metadata);
        } else {
            this.metadata = null;
        }
    }

    /**
     * Construct an empty example using the supplied output and weight.
     * @param output The output.
     * @param weight The weight.
     */
    protected Example(T output, float weight) {
        this.output = output;
        this.weight = weight;
    }

    /**
     * Construct an empty example using the supplied output, metadata and
     * {@link Example#DEFAULT_WEIGHT} as the weight.
     * @param output The output.
     * @param metadata The metadata.
     */
    protected Example(T output, Map<String,Object> metadata) {
        this.output = output;
        if (metadata != null && !metadata.isEmpty()) {
            this.metadata = new HashMap<>(metadata);
        } else {
            this.metadata = null;
        }
    }

    /**
     * Construct an empty example using the supplied output and
     * {@link Example#DEFAULT_WEIGHT} as the weight.
     * @param output The output.
     */
    protected Example(T output) {
        this.output = output;
    }

    /**
     * Copies the output, weight and metadata into this example.
     * @param other The example to copy.
     */
    protected Example(Example<T> other) {
        this.output = other.output;
        this.weight = other.weight;
        if (other.metadata != null && !other.metadata.isEmpty()) {
            this.metadata = new HashMap<>(other.metadata);
        } else {
            this.metadata = null;
        }
    }

    /**
     * Gets the example's {@link Output}.
     * @return The example's output.
     */
    public T getOutput() {
        return output;
    }

    /**
     * Gets the example's weight.
     * @return The example's weight.
     */
    public float getWeight() {
        return weight;
    }

    /**
     * Sets the example's weight.
     * @param weight The new weight.
     */
    public void setWeight(float weight) {
        this.weight = weight;
    }

    /**
     * Gets the associated metadata value for this key, if it exists.
     * Otherwise return {@link Optional#empty()}.
     * @param key The key to check.
     * @return The value if present.
     */
    public synchronized Optional<Object> getMetadataValue(String key) {
        if (metadata != null) {
            return Optional.ofNullable(metadata.get(key));
        } else {
            return Optional.empty();
        }
    }

    /**
     * Puts the specified key, value pair into the metadata.
     * <p>
     * Example metadata is append only, and so this method
     * throws {@link IllegalArgumentException} if the key is already present.
     * @param key The key.
     * @param value The value.
     */
    public synchronized void setMetadataValue(String key, Object value) {
        if (containsMetadata(key)) {
            Object oldValue = metadata.get(key);
            throw new IllegalArgumentException("Example metadata is append only. Key '" + key + "' is already associated with value '" + oldValue + "'");
        }
        if (metadata == null) {
            metadata = new HashMap<>(DEFAULT_METADATA_SIZE);
        }
        metadata.put(key,value);
    }

    /**
     * Test if the metadata contains the supplied key.
     * @param key The key to test.
     * @return True if the metadata contains a value for the supplied key.
     */
    public boolean containsMetadata(String key) {
        if (metadata != null) {
            return metadata.containsKey(key);
        } else {
            return false;
        }
    }

    /**
     * Returns a copy of this example's metadata.
     * @return The metadata.
     */
    public Map<String,Object> getMetadata() {
        if (metadata != null) {
            return new HashMap<>(metadata);
        } else {
            return Collections.emptyMap();
        }
    }

    /**
     * Sorts the example by the string comparator.
     */
    protected abstract void sort();

    /**
     * Adds a feature. This maintains the sorted invariant and has a lg(example.size())
     * cost per insertion.
     * @param feature The feature to add.
     */
    public abstract void add(Feature feature);

    /**
     * Adds a collection of features. This maintains the sorted invariant but is
     * more efficient than adding one at a time due to allocation.
     * @param features The features to add.
     */
    public abstract void addAll(Collection<? extends Feature> features);

    /**
     * Return how many features are in this example.
     * @return The number of features.
     */
    public abstract int size();

    /**
     * Removes all features in this list from the Example.
     * @param featureList Features to remove from this Example.
     */
    public abstract void removeFeatures(List<Feature> featureList);

    /**
     * Merges features with the same name using the
     * supplied {@link Merger}.
     * @param merger A function to merge two doubles.
     */
    public abstract void reduceByName(Merger merger);

    /**
     * Checks the example to see if all the feature names are unique,
     * the feature values are not NaN, and there is at least one feature.
     * @return true if the example is valid.
     */
    public abstract boolean validateExample();

    /**
     * Transforms this example by applying the transformations from the supplied {@link TransformerMap}.
     * <p>
     * Can be overridden for performance reasons.
     * @param transformerMap The transformations to apply.
     */
    public void transform(TransformerMap transformerMap) {
        for (Map.Entry<String,List<Transformer>> e : transformerMap.entrySet()) {
            Feature f = lookup(e.getKey());
            if (f != null) {
                double value = f.getValue();
                for (Transformer t : e.getValue()) {
                    value = t.transform(value);
                }
                set(new Feature(f.getName(),value));
            }
        }
    }

    /**
     * Is this example dense wrt the supplied feature map.
     * <p>
     * An example is "dense" if it contains all the features in the map,
     * and only those features.
     * @param fMap The feature map to check against.
     * @return True if this example contains only the features in the map, and all the features in the map.
     */
    public abstract boolean isDense(FeatureMap fMap);

    /**
     * Converts all implicit zeros into explicit zeros based on the supplied feature map.
     * @param fMap The feature map to use for densification.
     */
    public void densify(FeatureMap fMap) {
        // Densify! - guitar solo
        List<String> featureNames = new ArrayList<>(fMap.keySet());
        Collections.sort(featureNames);
        densify(featureNames);
    }

    /**
     * Converts all implicit zeros into explicit zeros based on the supplied feature names.
     * @param featureNames A *sorted* list of feature names.
     */
    protected abstract void densify(List<String> featureNames);

    /**
     * Returns a deep copy of this Example.
     * @return A deep copy of this example.
     */
    public abstract Example<T> copy();

    /**
     * Returns the Feature in this Example which has the supplied name, if it's present.
     * @param i The feature name to lookup.
     * @return The feature object.
     */
    public Feature lookup(String i) {
        for (Feature f : this) {
            if (i.equals(f.getName())) {
                return f;
            }
        }
        return null;
    }

    /**
     * Overwrites the feature with the matching name.
     * <p>
     * Throws {@link IllegalArgumentException} if there isn't a feature with that
     * name in this example.
     * @param feature The new feature value.
     */
    public abstract void set(Feature feature);

    /**
     * Reassigns feature name Strings in the Example to point to those in the {@link FeatureMap}.
     * This significantly reduces memory allocation. It is called when an Example is added
     * to a {@link MutableDataset}, and should not be called outside of that context as it may interact
     * unexpectedly with {@link HashedFeatureMap}.
     * @param featureMap The feature map containing canonical feature names.
     */
    public abstract void canonicalize(FeatureMap featureMap);

    /**
     * Deserializes an example proto into an example.
     * @param proto The proto to deserialize.
     * @return The deserialized example.
     */
    public static Example<?> deserialize(ExampleProto proto) {
        return ProtoUtil.deserialize(proto);
    }
}
