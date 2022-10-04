/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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
import com.oracle.labs.mlrg.olcut.util.MutableNumber;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputInfo;
import org.tribuo.classification.Label;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoSerializableKeysValuesField;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The base class for information about {@link MultiLabel} outputs.
 */
public abstract class MultiLabelInfo implements OutputInfo<MultiLabel> {
    private static final long serialVersionUID = 1L;

    /**
     * The map of label occurrence counts.
     */
    @ProtoSerializableKeysValuesField(keysName="label",valuesName="count")
    protected final Map<String,MutableLong> labelCounts;
    /**
     * The number of times the unknown label has been observed.
     */
    @ProtoSerializableField
    protected int unknownCount = 0;
    /**
     * The label domain.
     */
    protected transient Map<String,MultiLabel> labels;

    /**
     * The total number of {@link MultiLabel} objects this object has seen.
     */
    @ProtoSerializableField
    protected int totalCount = 0;

    /**
     * Construct a MultiLabelInfo, initializing the various count variables.
     */
    MultiLabelInfo() {
        labelCounts = new HashMap<>();
        labels = new HashMap<>();
    }

    /**
     * Copy the MultiLabelInfo. The copy ignores the unknown count.
     * @param other The MultiLabelInfo to copy.
     */
    MultiLabelInfo(MultiLabelInfo other) {
        labelCounts = MutableNumber.copyMap(other.labelCounts);
        labels = new HashMap<>(other.labels);
        totalCount = other.totalCount;
    }

    /**
     * Deserialization constructor.
     * @param counts Counts map.
     * @param unknownCount Unknown count.
     * @param totalCount Total count.
     */
    MultiLabelInfo(Map<String,MutableLong> counts, int unknownCount, int totalCount) {
        if (unknownCount < 0) {
            throw new IllegalArgumentException("Unknown count must be non-negative, found " + unknownCount);
        }
        if (totalCount < 0) {
            throw new IllegalArgumentException("Total count must be non-negative, found " + totalCount);
        }
        this.unknownCount = unknownCount;
        this.totalCount = totalCount;
        labelCounts = new HashMap<>();
        labels = new HashMap<>();
        for (Map.Entry<String,MutableLong> e : counts.entrySet()) {
            if (e.getValue().longValue() < 1) {
                throw new IllegalArgumentException("Count for " + e.getKey() + " must be positive but found " + e.getValue().longValue());
            }
            labelCounts.put(e.getKey(),e.getValue().copy());
            labels.put(e.getKey(), new MultiLabel(e.getKey()));
        }
    }

    @Override
    public int getUnknownCount() {
        return unknownCount;
    }

    /**
     * Returns a set of MultiLabel, where each has a single Label inside it.
     * The set covers the space of Labels that this MultiLabelInfo has seen.
     * @return The set of possible labels.
     */
    @Override
    public Set<MultiLabel> getDomain() {
        return new HashSet<>(labels.values());
    }

    /**
     * Get the number of times this Label was observed, or 0 if unknown.
     * @param label The Label to look for.
     * @return A non-negative long.
     */
    public long getLabelCount(Label label) {
        MutableLong l = labelCounts.get(label.getLabel());
        if (l != null) {
            return l.longValue();
        } else {
            return 0;
        }
    }

    /**
     * Get the number of times this String was observed, or 0 if unknown.
     * @param label The String to look for.
     * @return A non-negative long.
     */
    public long getLabelCount(String label) {
        MutableLong l = labelCounts.get(label);
        if (l != null) {
            return l.longValue();
        } else {
            return 0;
        }
    }

    @Override
    public Iterable<Pair<String,Long>> outputCountsIterable() {
        return () -> new Iterator<Pair<String, Long>>() {
            Iterator<Map.Entry<String, MutableLong>> itr = labelCounts.entrySet().iterator();

            @Override
            public boolean hasNext() {
                return itr.hasNext();
            }

            @Override
            public Pair<String, Long> next() {
                Map.Entry<String, MutableLong> e = itr.next();
                return new Pair<>(e.getKey(), e.getValue().longValue());
            }
        };
    }

    @Override
    public int size() {
        return labelCounts.size();
    }

    @Override
    public ImmutableOutputInfo<MultiLabel> generateImmutableOutputInfo() {
        return new ImmutableMultiLabelInfo(this);
    }

    @Override
    public MutableOutputInfo<MultiLabel> generateMutableOutputInfo() {
        return new MutableMultiLabelInfo(this);
    }

    @Override
    public abstract MultiLabelInfo copy();

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MultiLabelInfo that = (MultiLabelInfo) o;
        if (unknownCount == that.unknownCount && totalCount == that.totalCount) {
            for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
                MutableLong other = that.labelCounts.get(e.getKey());
                if (other == null || (other.longValue() != e.getValue().longValue())) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        return Objects.hash(labelCounts, unknownCount, totalCount);
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        labels = new HashMap<>();
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            labels.put(e.getKey(),new MultiLabel(e.getKey()));
        }
    }
}
