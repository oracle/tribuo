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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputInfo;
import org.tribuo.protos.core.OutputDomainProto;
import org.tribuo.test.protos.MockOutputInfoProto;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
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
        labelCounts = new HashMap<>(other.labelCounts);
        labelCounter = other.labelCounter;
        unknownCount = other.unknownCount;
        labels = new HashMap<>(other.labels);
        idLabelMap = new HashMap<>(other.idLabelMap);
        labelIDMap = new HashMap<>(other.labelIDMap);
    }

    private MockOutputInfo(Map<String, MutableLong> labelCounts, Map<String, Integer> labelIDMap, int unknownCount, int labelCounter) {
        this.unknownCount = unknownCount;
        this.labelCounter = labelCounter;
        this.labelCounts = labelCounts;
        this.labelIDMap = labelIDMap;
        this.idLabelMap = new HashMap<>();
        this.labels = new HashMap<>();
        for (Map.Entry<String,Integer> e : labelIDMap.entrySet()) {
            idLabelMap.put(e.getValue(),e.getKey());
            labels.put(e.getKey(),new MockOutput(e.getKey()));
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
    public static MockOutputInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
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

        return new MockOutputInfo(counts, ids, proto.getUnknownCount(), proto.getLabelCounter());
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
        outputBuilder.setClassName(MockOutputInfo.class.getName());
        outputBuilder.setSerializedData(Any.pack(protoBuilder.build()));

        return outputBuilder.build();
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
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MockOutputInfo pairs = (MockOutputInfo) o;
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            MutableLong other = pairs.labelCounts.get(e.getKey());
            if (other == null || (other.longValue() != e.getValue().longValue())) {
                return false;
            }
        }
        return unknownCount == pairs.unknownCount && labelCounter == pairs.labelCounter && labels.equals(pairs.labels) && idLabelMap.equals(pairs.idLabelMap) && labelIDMap.equals(pairs.labelIDMap);
    }

    @Override
    public int hashCode() {
        return Objects.hash(unknownCount, labelCounter, labels, idLabelMap, labelIDMap);
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
