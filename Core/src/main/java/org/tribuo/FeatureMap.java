/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.protos.core.FeatureDomainProto;
import org.tribuo.util.ProtoUtil;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;

/**
 * A map from Strings to {@link VariableInfo} objects storing
 * information about a feature.
 */
public abstract class FeatureMap implements Serializable, ProtoSerializable<FeatureDomainProto>, Iterable<VariableInfo> {
    private static final long serialVersionUID = 1L;

    /**
     * Map from the feature names to their info.
     */
    @ProtoSerializableField(name="info")
    protected final Map<String, VariableInfo> m;

    /**
     * Constructs an empty feature map.
     */
    protected FeatureMap() {
        m = new HashMap<>();
    }

    /**
     * Constructs a deep copy of the supplied feature map.
     * @param map The map to copy.
     */
    protected FeatureMap(FeatureMap map) {
        m = new HashMap<>();
        for (Map.Entry<String,VariableInfo> e : map.m.entrySet()) {
            VariableInfo info = e.getValue().copy();

            m.put(e.getKey(),info);
        }
    }

    /**
     * Constructs a feature map wrapping the supplied map.
     * <p>
     * Note the map is not defensively copied.
     * @param m The map to wrap.
     */
    @SuppressWarnings("unchecked") // upcasting off the wildcard.
    protected FeatureMap(Map<String, ? extends VariableInfo> m) {
        this.m = (Map<String,VariableInfo>) m;
    }

    /**
     * Gets the variable info associated with that feature name, or null if it's unknown.
     * @param name The feature name.
     * @return The variable info or null.
     */
    public VariableInfo get(String name) {
        return m.get(name);
    }

    /**
     * Returns the number of features in the domain.
     * @return The number of features.
     */
    public int size() {
        return m.size();
    }

    @Override
    public String toString() {
        return m.toString();
    }

    /**
     * Returns all the feature names in the domain.
     * @return The feature names.
     */
    public Set<String> keySet() {
        return m.keySet();
    }

    @Override
    public Iterator<VariableInfo> iterator() {
        return m.values().iterator();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        FeatureMap that = (FeatureMap) o;
        return m.equals(that.m);
    }

    @Override
    public int hashCode() {
        return Objects.hash(m);
    }

    /**
     * Same as the toString, but ordered by name, and with newlines.
     * @return A String representation of this FeatureMap.
     */
    public String toReadableString() {
        StringBuilder sb = new StringBuilder();
        TreeMap<String,VariableInfo> tm = new TreeMap<>(m);
        for (Map.Entry<String, VariableInfo> e : tm.entrySet()) {
            if(sb.length() > 0) {
                sb.append('\n');
            }
            sb.append(e.getValue().toString());
        }
        return sb.toString();
    }

    /**
     * Check if this feature map contains the same features as the supplied one.
     * @param other The feature map to check.
     * @return True if the two feature maps contain the same named features.
     */
    public boolean domainEquals(FeatureMap other) {
        if (size() == other.size()) {
            for (Map.Entry<String, VariableInfo> e : m.entrySet()) {
                VariableInfo otherInfo = other.get(e.getKey());
                if (otherInfo == null) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    /**
     * Deserializes a {@link FeatureDomainProto} into a {@link FeatureMap} subclass.
     * @param proto The proto to deserialize.
     * @return The deserialized FeatureMap.
     */
    public static FeatureMap deserialize(FeatureDomainProto proto) {
        return (FeatureMap) ProtoUtil.instantiate(proto.getVersion(), proto.getClassName(), proto.getSerializedData());
    }

}
