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

package org.tribuo.classification;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.MutableNumber;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputInfo;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoSerializableKeysValuesField;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * The base class for information about multi-class classification Labels.
 */
public abstract class LabelInfo implements OutputInfo<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * The occurrence counts of each label.
     */
    @ProtoSerializableKeysValuesField(keysName="label",valuesName="count")
    protected final Map<String,MutableLong> labelCounts;
    /**
     * The number of unknown labels this LabelInfo has seen.
     */
    @ProtoSerializableField
    protected int unknownCount = 0;
    /**
     * The label domain.
     */
    protected transient Map<String,Label> labels;

    /**
     * Constructs an empty label info.
     */
    LabelInfo() {
        labelCounts = new HashMap<>();
        labels = new HashMap<>();
    }

    /**
     * Copies the label info apart from the unknown count.
     * @param other The label info to copy.
     */
    LabelInfo(LabelInfo other) {
        labelCounts = MutableNumber.copyMap(other.labelCounts);
        labels = new HashMap<>();
        labels.putAll(other.labels);
    }

    /**
     * Deserialization constructor.
     * @param counts Counts map.
     * @param unknownCount Unknown count.
     */
    LabelInfo(Map<String,MutableLong> counts, int unknownCount) {
        if (unknownCount < 0) {
            throw new IllegalArgumentException("Unknown count must be non-negative, found " + unknownCount);
        }
        this.unknownCount = unknownCount;
        labelCounts = new HashMap<>();
        labels = new HashMap<>();
        for (Map.Entry<String,MutableLong> e : counts.entrySet()) {
            if (e.getValue().longValue() < 1) {
                throw new IllegalArgumentException("Count for " + e.getKey() + " must be positive but found " + e.getValue().longValue());
            }
            labelCounts.put(e.getKey(),e.getValue().copy());
            labels.put(e.getKey(), new Label(e.getKey()));
        }
    }

    @Override
    public int getUnknownCount() {
        return unknownCount;
    }

    /**
     * Returns the set of possible {@link Label}s that this LabelInfo has seen.
     * <p>
     * Each label has the default score of Double.NaN.
     * @return The set of possible labels.
     */
    @Override
    public Set<Label> getDomain() {
        return new HashSet<>(labels.values());
    }

    /**
     * Gets the count of the supplied label, or 0 if the label is unknown.
     * @param label A Label.
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
     * Gets the count of the supplied label, or 0 if the label is unknown.
     * @param label A String representing a Label.
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
        return () -> new Iterator<Pair<String,Long>>() {
            Iterator<Map.Entry<String,MutableLong>> itr = labelCounts.entrySet().iterator();

            @Override
            public boolean hasNext() {
                return itr.hasNext();
            }

            @Override
            public Pair<String,Long> next() {
                Map.Entry<String,MutableLong> e = itr.next();
                return new Pair<>(e.getKey(),e.getValue().longValue());
            }
        };
    }

    /**
     * The number of unique {@link Label}s this LabelInfo has seen.
     * @return The number of unique labels.
     */
    @Override
    public int size() {
        return labelCounts.size();
    }

    @Override
    public ImmutableOutputInfo<Label> generateImmutableOutputInfo() {
        return new ImmutableLabelInfo(this);
    }

    @Override
    public MutableOutputInfo<Label> generateMutableOutputInfo() {
        return new MutableLabelInfo(this);
    }

    @Override
    public abstract LabelInfo copy();

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        labels = new HashMap<>();
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            labels.put(e.getKey(),new Label(e.getKey()));
        }
    }
}
