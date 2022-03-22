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

package org.tribuo.multilabel;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An {@link ImmutableOutputInfo} for working with {@link MultiLabel} tasks.
 */
public class ImmutableMultiLabelInfo extends MultiLabelInfo implements ImmutableOutputInfo<MultiLabel> {
    private static final Logger logger = Logger.getLogger(ImmutableMultiLabelInfo.class.getName());

    private static final long serialVersionUID = 1L;

    private final Map<Integer,String> idLabelMap;

    private final Map<String,Integer> labelIDMap;

    private transient Set<MultiLabel> domain;

    private ImmutableMultiLabelInfo(ImmutableMultiLabelInfo info) {
        super(info);
        idLabelMap = new HashMap<>();
        idLabelMap.putAll(info.idLabelMap);
        labelIDMap = new HashMap<>();
        labelIDMap.putAll(info.labelIDMap);

        domain = Collections.unmodifiableSet(new HashSet<>(labels.values()));
    }

    ImmutableMultiLabelInfo(MultiLabelInfo info) {
        super(info);
        idLabelMap = new HashMap<>();
        labelIDMap = new HashMap<>();
        int counter = 0;
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            idLabelMap.put(counter,e.getKey());
            labelIDMap.put(e.getKey(),counter);
            counter++;
        }

        domain = Collections.unmodifiableSet(new HashSet<>(labels.values()));
    }

    ImmutableMultiLabelInfo(MutableMultiLabelInfo info, Map<MultiLabel, Integer> mapping) {
        super(info);
        if (mapping.size() != info.size()) {
            throw new IllegalStateException("Mapping and info come from different sources, mapping.size() = " + mapping.size() + ", info.size() = " + info.size());
        }

        idLabelMap = new HashMap<>();
        labelIDMap = new HashMap<>();
        for (Map.Entry<MultiLabel,Integer> e : mapping.entrySet()) {
            MultiLabel ml = e.getKey();
            Set<String> names = ml.getNameSet();
            if (names.size() == 1) {
                String name = names.iterator().next();
                idLabelMap.put(e.getValue(), name);
                labelIDMap.put(name, e.getValue());
            } else {
                throw new IllegalArgumentException("Mapping must contain a single label per id, but contains " + names + " -> " + e.getValue());
            }
        }

        domain = Collections.unmodifiableSet(new HashSet<>(labels.values()));
    }

    @Override
    public Set<MultiLabel> getDomain() {
        return domain;
    }

    @Override
    public int getID(MultiLabel output) {
        return labelIDMap.getOrDefault(output.getLabelString(), -1);
    }

    @Override
    public MultiLabel getOutput(int id) {
        String label = idLabelMap.get(id);
        if (label != null) {
            return labels.get(label);
        } else {
            logger.log(Level.INFO, "No entry found for id " + id);
            return null;
        }
    }

    @Override
    public long getTotalObservations() {
        return totalCount;
    }

    /**
     * Gets the id for the supplied label string.
     * @param label The label string.
     * @return The id number, or -1 if the label is unknown.
     */
    public int getID(String label) {
        return labelIDMap.getOrDefault(label,-1);
    }

    /**
     * Gets the count of the label occurrence for the specified id number, or 0 if it's unknown.
     * @param id The label id.
     * @return The label count.
     */
    public long getLabelCount(int id) {
        String label = idLabelMap.get(id);
        if (label != null) {
            MutableLong l = labelCounts.get(label);
            return l.longValue();
        } else {
            return 0;
        }
    }

    @Override
    public ImmutableMultiLabelInfo copy() {
        return new ImmutableMultiLabelInfo(this);
    }

    @Override
    public String toReadableString() {
        StringBuilder builder = new StringBuilder();
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            if (builder.length() > 0) {
                builder.append(", ");
            }
            builder.append('(');
            builder.append(labelIDMap.get(e.getKey()));
            builder.append(',');
            builder.append(e.getKey());
            builder.append(',');
            builder.append(e.getValue().longValue());
            builder.append(')');
        }
        return builder.toString();
    }

    @Override
    public String toString() {
        return toReadableString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ImmutableMultiLabelInfo pairs = (ImmutableMultiLabelInfo) o;
        return super.equals(o) &&
                idLabelMap.equals(pairs.idLabelMap) &&
                labelIDMap.equals(pairs.labelIDMap);
    }

    @Override
    public int hashCode() {
        return Objects.hash(labelCounts, idLabelMap, labelIDMap);
    }

    @Override
    public Iterator<Pair<Integer, MultiLabel>> iterator() {
        return new ImmutableInfoIterator(idLabelMap);
    }

    @Override
    public boolean domainAndIDEquals(ImmutableOutputInfo<MultiLabel> other) {
        if (size() == other.size()) {
            for (Map.Entry<Integer,String> e : idLabelMap.entrySet()) {
                MultiLabel otherLbl = other.getOutput(e.getKey());
                if (otherLbl == null) {
                    return false;
                } else if (!otherLbl.getLabelString().equals(e.getValue())) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    private static class ImmutableInfoIterator implements Iterator<Pair<Integer,MultiLabel>> {

        private final Iterator<Map.Entry<Integer,String>> itr;

        public ImmutableInfoIterator(Map<Integer,String> idLabelMap) {
            itr = idLabelMap.entrySet().iterator();
        }

        @Override
        public boolean hasNext() {
            return itr.hasNext();
        }

        @Override
        public Pair<Integer, MultiLabel> next() {
            Map.Entry<Integer,String> e = itr.next();
            return new Pair<>(e.getKey(),new MultiLabel(e.getValue()));
        }
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        domain = Collections.unmodifiableSet(new HashSet<>(labels.values()));
    }
}
