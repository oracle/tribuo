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

package org.tribuo.test;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.MutableNumber;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputInfo;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public abstract class MockMultiOutputInfo implements OutputInfo<MockMultiOutput> {

    protected final Map<String, MutableLong> labelCounts;
    protected int unknownCount = 0;
    protected transient Map<String,MockMultiOutput> labels;

    protected int totalCount = 0;

    public MockMultiOutputInfo() {
        labelCounts = new HashMap<>();
        labels = new HashMap<>();
    }

    public MockMultiOutputInfo(MockMultiOutputInfo other) {
        labelCounts = MutableNumber.copyMap(other.labelCounts);
        labels = new HashMap<>(other.labels);
        totalCount = other.totalCount;
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
    public Set<MockMultiOutput> getDomain() {
        return new HashSet<>(labels.values());
    }

    /**
     * Get the number of times this Label was observed, or 0 if unknown.
     * @param label The Label to look for.
     * @return A non-negative long.
     */
    public long getLabelCount(MockOutput label) {
        MutableLong l = labelCounts.get(label.label);
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
    public ImmutableOutputInfo<MockMultiOutput> generateImmutableOutputInfo() {
        return new ImmutableMockMultiOutputInfo(this);
    }

    @Override
    public MutableOutputInfo<MockMultiOutput> generateMutableOutputInfo() {
        return new MutableMockMultiOutputInfo(this);
    }

    @Override
    public abstract MockMultiOutputInfo copy();

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        labels = new HashMap<>();
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            labels.put(e.getKey(),new MockMultiOutput(e.getKey()));
        }
    }

}