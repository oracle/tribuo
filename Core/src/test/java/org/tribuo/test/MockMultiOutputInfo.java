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

package org.tribuo.test;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.MutableNumber;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputInfo;
import org.tribuo.protos.core.OutputDomainProto;
import org.tribuo.test.protos.MockOutputInfoProto;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * An OutputInfo for use in tests, very similar to MultiLabelInfo.
 * <p>
 * Implements both MutableOutputInfo and ImmutableOutputInfo. Don't do this in real code!
 */
public class MockMultiOutputInfo implements OutputInfo<MockMultiOutput>, ImmutableOutputInfo<MockMultiOutput>, MutableOutputInfo<MockMultiOutput>  {

    protected final Map<String, MutableLong> labelCounts;
    protected int unknownCount = 0;
    protected transient Map<String,MockMultiOutput> labels;

    protected int totalCount = 0;

    private int labelCounter = 0;

    private Map<Integer,String> idLabelMap;

    private Map<String,Integer> labelIDMap;

    public MockMultiOutputInfo() {
        labelCounts = new HashMap<>();
        labels = new HashMap<>();
        idLabelMap = new HashMap<>();
        labelIDMap = new HashMap<>();
    }

    private MockMultiOutputInfo(MockMultiOutputInfo other) {
        labelCounts = MutableNumber.copyMap(other.labelCounts);
        labelCounter = other.labelCounter;
        labels = new HashMap<>(other.labels);
        totalCount = other.totalCount;
        idLabelMap = new HashMap<>();
        idLabelMap.putAll(other.idLabelMap);
        labelIDMap = new HashMap<>();
        labelIDMap.putAll(other.labelIDMap);
    }

    private MockMultiOutputInfo(Map<String, MutableLong> labelCounts, Map<String, Integer> labelIDMap, int unknownCount, int labelCounter) {
        this.unknownCount = unknownCount;
        this.labelCounter = labelCounter;
        this.labelCounts = labelCounts;
        this.labelIDMap = labelIDMap;
        this.idLabelMap = new HashMap<>();
        this.labels = new HashMap<>();
        for (Map.Entry<String,Integer> e : labelIDMap.entrySet()) {
            idLabelMap.put(e.getValue(),e.getKey());
            labels.put(e.getKey(),new MockMultiOutput(e.getKey()));
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
    public static MockMultiOutputInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        MockOutputInfoProto proto = message.unpack(MockOutputInfoProto.class);

        if (proto.getLabelCount() != proto.getCountsCount() || proto.getCountsCount() != proto.getIdCount()) {
            throw new IllegalArgumentException("Invalid protobuf, must have a label, id and count for each entry");
        }
        HashMap<String, MutableLong> counts = new HashMap<>();
        HashMap<String, Integer> ids = new HashMap<>();
        for (int i = 0; i < proto.getLabelCount(); i++) {
            MutableLong count = new MutableLong(proto.getCounts(i));
            if (count.longValue() < 1) {
                throw new IllegalArgumentException("Counts must be positive, for label " + proto.getLabel(i) + " found " + count);
            }
            counts.put(proto.getLabel(i), count);
            int tmpId = proto.getId(i);
            if (tmpId < 0) {
                throw new IllegalArgumentException("Id must be non-negative, for label " + proto.getLabel(i) + " found " + tmpId);
            }
            ids.put(proto.getLabel(i),tmpId);
        }

        return new MockMultiOutputInfo(counts, ids, proto.getUnknownCount(), proto.getLabelCounter());
    }

    @Override
    public OutputDomainProto serialize() {
        MockOutputInfoProto.Builder protoBuilder = MockOutputInfoProto.newBuilder();

        protoBuilder.setLabelCounter(labelCounter);
        protoBuilder.setUnknownCount(unknownCount);
        for (Map.Entry<String, MutableLong> e : labelCounts.entrySet()) {
            protoBuilder.addLabel(e.getKey());
            protoBuilder.addCounts(e.getValue().longValue());
            protoBuilder.addId(labelIDMap.get(e.getKey()));
        }

        OutputDomainProto.Builder outputBuilder = OutputDomainProto.newBuilder();

        outputBuilder.setVersion(0);
        outputBuilder.setClassName(this.getClass().getName());
        outputBuilder.setSerializedData(Any.pack(protoBuilder.build()));

        return outputBuilder.build();
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

    /**
     * Throws IllegalStateException if the MockMultiOutput contains a Label which has a "," in it.
     *
     * Such labels are disallowed. There should be an exception thrown when one is constructed
     * too.
     * @param output The observed output.
     */
    @Override
    public void observe(MockMultiOutput output) {
        if (output == MockMultiOutputFactory.UNKNOWN_MULTILABEL) {
            unknownCount++;
        } else {
            for (String label : output.getNameSet()) {
                if (label.contains(",")) {
                    throw new IllegalStateException("MockMultiOutput cannot use a Label which contains ','. The supplied label was " + label + ".");
                }
                MutableLong value = labelCounts.computeIfAbsent(label, k -> new MutableLong());
                labels.computeIfAbsent(label, MockMultiOutput::new);
                value.increment();
                if (!labelIDMap.containsKey(label)) {
                    labelIDMap.put(label,labelCounter);
                    idLabelMap.put(labelCounter,label);
                    labelCounter++;
                }
            }
            totalCount++;
        }
    }

    @Override
    public void clear() {
        labelCounts.clear();
        totalCount = 0;
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
        return new MockMultiOutputInfo(this);
    }

    @Override
    public MutableOutputInfo<MockMultiOutput> generateMutableOutputInfo() {
        return new MockMultiOutputInfo(this);
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
        MockMultiOutputInfo otherInfo = (MockMultiOutputInfo) other;
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
    public MockMultiOutputInfo copy() {
        return new MockMultiOutputInfo(this);
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

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        labels = new HashMap<>();
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            labels.put(e.getKey(),new MockMultiOutput(e.getKey()));
        }
    }

}