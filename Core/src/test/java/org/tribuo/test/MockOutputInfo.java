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

package org.tribuo.test;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputInfo;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An OutputInfo for use in tests, very similar to LabelInfo.
 * <p>
 * Implements both MutableOutputInfo and ImmutableOutputInfo. Don't do this in real code!
 */
public class MockOutputInfo implements MutableOutputInfo<MockOutput>, ImmutableOutputInfo<MockOutput> {
    private static final Logger logger = Logger.getLogger(MockOutputInfo.class.getName());

    private final Map<String, MutableLong> labelCounts;
    private int unknownCount = 0;
    private int labelCounter = 0;
    private final Map<String,MockOutput> labels;

    private final Map<Integer,String> idLabelMap;

    private final Map<String,Integer> labelIDMap;

    public MockOutputInfo() {
        labelCounts = new HashMap<>();
        labels = new HashMap<>();
        idLabelMap = new HashMap<>();
        labelIDMap = new HashMap<>();
    }

    private MockOutputInfo(MockOutputInfo other) {
        labelCounts = new HashMap<>(labelCounter);
        labelCounter = other.labelCounter;
        unknownCount = other.unknownCount;
        labels = new HashMap<>(other.labels);
        idLabelMap = new HashMap<>(other.idLabelMap);
        labelIDMap = new HashMap<>(other.labelIDMap);
    }

    @Override
    public void observe(MockOutput output) {
        if (output == MockOutputFactory.UNKNOWN_TEST_OUTPUT) {
            unknownCount++;
        } else {
            String label = output.label;
            MutableLong value = labelCounts.computeIfAbsent(label, k -> new MutableLong());
            labels.computeIfAbsent(label, MockOutput::new);
            value.increment();
            if (!labelIDMap.containsKey(label)) {
                labelIDMap.put(label,labelCounter);
                idLabelMap.put(labelCounter,label);
                labelCounter++;
            }
        }
    }

    @Override
    public Set<MockOutput> getDomain() { return new HashSet<>(labels.values()); }

    @Override
    public int size() { return labelCounts.size(); }

    @Override
    public int getUnknownCount() {
        return unknownCount;
    }

    @Override
    public ImmutableOutputInfo<MockOutput> generateImmutableOutputInfo() {
        return new MockOutputInfo(this);
    }

    @Override
    public MutableOutputInfo<MockOutput> generateMutableOutputInfo() {
        return new MockOutputInfo(this);
    }

    @Override
    public OutputInfo<MockOutput> copy() {
        return new MockOutputInfo(this);
    }

    @Override
    public void clear() {
        labelCounts.clear();
        idLabelMap.clear();
        labelIDMap.clear();
        labelCounter = 0;
    }

    @Override
    public String toReadableString() { return labelCounts.toString(); }

    @Override
    public Iterable<Pair<String, Long>> outputCountsIterable() {
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

    @Override
    public int getID(MockOutput output) {
        return labelIDMap.getOrDefault(output.label,-1);
    }

    @Override
    public MockOutput getOutput(int id) {
        String label = idLabelMap.get(id);
        if (label != null) {
            return labels.get(label);
        } else {
            logger.log(Level.INFO,"No entry found for id " + id);
            return null;
        }
    }

    @Override
    public long getTotalObservations() {
        long count = 0;
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            count += e.getValue().longValue();
        }
        return count;
    }

    @Override
    public boolean domainAndIDEquals(ImmutableOutputInfo<MockOutput> other) {
        MockOutputInfo otherInfo = (MockOutputInfo) other;
        return otherInfo.idLabelMap.equals(idLabelMap);
    }

    @Override
    public Iterator<Pair<Integer, MockOutput>> iterator() {
        return new ImmutableInfoIterator(idLabelMap);
    }

    private static class ImmutableInfoIterator implements Iterator<Pair<Integer,MockOutput>> {

        private final Iterator<Map.Entry<Integer,String>> itr;

        public ImmutableInfoIterator(Map<Integer,String> idLabelMap) {
            itr = idLabelMap.entrySet().iterator();
        }

        @Override
        public boolean hasNext() {
            return itr.hasNext();
        }

        @Override
        public Pair<Integer, MockOutput> next() {
            Map.Entry<Integer,String> e = itr.next();
            return new Pair<>(e.getKey(),new MockOutput(e.getValue()));
        }
    }
}
