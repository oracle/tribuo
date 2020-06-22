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

package org.tribuo;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * A map from Strings to {@link VariableInfo} objects storing
 * information about a feature.
 */
public abstract class FeatureMap implements Serializable, Iterable<VariableInfo> {
    private static final long serialVersionUID = 1L;

    protected final Map<String, VariableInfo> m;

    protected FeatureMap() {
        m = new HashMap<>();
    }

    protected FeatureMap(FeatureMap map) {
        m = new HashMap<>();
        for (Map.Entry<String,VariableInfo> e : map.m.entrySet()) {
            VariableInfo info = e.getValue().copy();

            m.put(e.getKey(),info);
        }
    }

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

}
