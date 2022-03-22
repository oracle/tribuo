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

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class ImmutableMockMultiOutputInfo extends MockMultiOutputInfo implements ImmutableOutputInfo<MockMultiOutput> {

    private Map<Integer,String> idLabelMap;

    private Map<String,Integer> labelIDMap;

    public ImmutableMockMultiOutputInfo(ImmutableMockMultiOutputInfo info) {
        super(info);
        idLabelMap = new HashMap<>();
        idLabelMap.putAll(info.idLabelMap);
        labelIDMap = new HashMap<>();
        labelIDMap.putAll(info.labelIDMap);
    }

    public ImmutableMockMultiOutputInfo(MockMultiOutputInfo info) {
        super(info);
        idLabelMap = new HashMap<>();
        labelIDMap = new HashMap<>();
        int counter = 0;
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            idLabelMap.put(counter,e.getKey());
            labelIDMap.put(e.getKey(),counter);
            counter++;
        }
    }

    @Override
    public int getID(MockMultiOutput output) {
        return labelIDMap.getOrDefault(output.getLabelString(), -1);
    }

    @Override
    public MockMultiOutput getOutput(int id) {
        String label = idLabelMap.get(id);
        if (label != null) {
            return labels.get(label);
        } else {
            return null;
        }
    }

    @Override
    public long getTotalObservations() {
        return totalCount;
    }

    @Override
    public boolean domainAndIDEquals(ImmutableOutputInfo<MockMultiOutput> other) {
        ImmutableMockMultiOutputInfo otherInfo = (ImmutableMockMultiOutputInfo) other;
        return otherInfo.idLabelMap.equals(idLabelMap);
    }

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
    public ImmutableMockMultiOutputInfo copy() {
        return new ImmutableMockMultiOutputInfo(this);
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
    public Iterator<Pair<Integer, MockMultiOutput>> iterator() {
        return new ImmutableInfoIterator(idLabelMap);
    }

    private static class ImmutableInfoIterator implements Iterator<Pair<Integer,MockMultiOutput>> {

        private final Iterator<Map.Entry<Integer,String>> itr;

        public ImmutableInfoIterator(Map<Integer,String> idLabelMap) {
            itr = idLabelMap.entrySet().iterator();
        }

        @Override
        public boolean hasNext() {
            return itr.hasNext();
        }

        @Override
        public Pair<Integer, MockMultiOutput> next() {
            Map.Entry<Integer,String> e = itr.next();
            return new Pair<>(e.getKey(),new MockMultiOutput(e.getValue()));
        }
    }
}
