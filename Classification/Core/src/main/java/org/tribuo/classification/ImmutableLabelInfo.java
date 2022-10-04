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

package org.tribuo.classification;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.classification.protos.ImmutableLabelInfoProto;
import org.tribuo.protos.core.OutputDomainProto;

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
 * An {@link ImmutableOutputInfo} object for {@link Label}s.
 * <p>
 * Gives each unique label an id number. Also counts each label occurrence like {@link MutableLabelInfo} does,
 * though the counts are frozen in this object.
 */
public class ImmutableLabelInfo extends LabelInfo implements ImmutableOutputInfo<Label> {
    private static final Logger logger = Logger.getLogger(ImmutableLabelInfo.class.getName());

    private static final long serialVersionUID = 1L;

    private final Map<Integer,String> idLabelMap;

    private final Map<String,Integer> labelIDMap;

    private transient Set<Label> domain;

    private ImmutableLabelInfo(ImmutableLabelInfo info) {
        super(info);
        idLabelMap = new HashMap<>();
        idLabelMap.putAll(info.idLabelMap);
        labelIDMap = new HashMap<>();
        labelIDMap.putAll(info.labelIDMap);
        domain = Collections.unmodifiableSet(new HashSet<>(labels.values()));
    }

    ImmutableLabelInfo(LabelInfo info) {
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

    ImmutableLabelInfo(LabelInfo info, Map<Label,Integer> mapping) {
        super(info);
        if (mapping.size() != info.size()) {
            throw new IllegalStateException("Mapping and info come from different sources, mapping.size() = " + mapping.size() + ", info.size() = " + info.size());
        }

        idLabelMap = new HashMap<>();
        labelIDMap = new HashMap<>();
        for (Map.Entry<Label,Integer> e : mapping.entrySet()) {
            idLabelMap.put(e.getValue(),e.getKey().label);
            labelIDMap.put(e.getKey().label,e.getValue());
        }
        domain = Collections.unmodifiableSet(new HashSet<>(labels.values()));
    }

    /**
     * Deserialization constructor.
     * @param labelCounts Counts map.
     * @param mapping Label id mapping.
     * @param unknownCount Unknown count.
     */
    private ImmutableLabelInfo(Map<String,MutableLong> labelCounts, Map<String, Integer> mapping, int unknownCount) {
        super(labelCounts,unknownCount);
        this.idLabelMap = new HashMap<>();
        this.labelIDMap = new HashMap<>();
        for (Map.Entry<String,Integer> e : mapping.entrySet()) {
            this.idLabelMap.put(e.getValue(),e.getKey());
            this.labelIDMap.put(e.getKey(),e.getValue());
        }
        this.domain = Collections.unmodifiableSet(new HashSet<>(labels.values()));
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static ImmutableLabelInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        ImmutableLabelInfoProto proto = message.unpack(ImmutableLabelInfoProto.class);
        if ((proto.getLabelCount() != proto.getCountCount()) || (proto.getLabelCount() != proto.getIdCount())) {
            throw new IllegalArgumentException("Invalid protobuf, different numbers of labels, ids and counts, labels " + proto.getLabelCount() + ", ids " + proto.getIdCount() + ", counts " + proto.getCountCount());
        }
        Map<String,MutableLong> labelCounts = new HashMap<>();
        Map<String,Integer> labelIDMap = new HashMap<>();
        for (int i = 0; i < proto.getLabelCount(); i++) {
            String lbl = proto.getLabel(i);
            long cnt = proto.getCount(i);
            int id = proto.getId(i);
            MutableLong old = labelCounts.put(lbl,new MutableLong(cnt));
            if (old != null) {
                throw new IllegalArgumentException("Invalid protobuf, two mappings for " + lbl);
            }
            labelIDMap.put(lbl,id);
        }
        return new ImmutableLabelInfo(labelCounts,labelIDMap,proto.getUnknownCount());
    }

    @Override
    public OutputDomainProto serialize() {
        OutputDomainProto.Builder domainBuilder = OutputDomainProto.newBuilder();

        domainBuilder.setClassName(ImmutableLabelInfo.class.getName());
        domainBuilder.setVersion(0);

        ImmutableLabelInfoProto.Builder data = ImmutableLabelInfoProto.newBuilder();
        data.setUnknownCount(unknownCount);
        for (Map.Entry<String, MutableLong> e : labelCounts.entrySet()) {
            data.addLabel(e.getKey());
            data.addCount(e.getValue().longValue());
            data.addId(labelIDMap.get(e.getKey()));
        }

        domainBuilder.setSerializedData(Any.pack(data.build()));

        return domainBuilder.build();
    }

    /**
     * Returns the set of possible {@link Label}s that this LabelInfo has seen.
     *
     * Each label has the default score of Double.NaN.
     * @return The set of possible labels.
     */
    @Override
    public Set<Label> getDomain() {
        return domain;
    }

    @Override
    public int getID(Label output) {
        return labelIDMap.getOrDefault(output.getLabel(),-1);
    }

    @Override
    public Label getOutput(int id) {
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

    /**
     * Returns the number of times the supplied id was observed before this LabelInfo was frozen.
     * @param id The id number.
     * @return The count.
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
    public ImmutableLabelInfo copy() {
        return new ImmutableLabelInfo(this);
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
        ImmutableLabelInfo labelInfo = (ImmutableLabelInfo) o;
        if (unknownCount == labelInfo.unknownCount && idLabelMap.equals(labelInfo.idLabelMap) && labelCounts.size() == labelInfo.labelCounts.size()) {
            for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
                MutableLong other = labelInfo.labelCounts.get(e.getKey());
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
        return Objects.hash(idLabelMap,labelCounts,unknownCount);
    }

    @Override
    public Iterator<Pair<Integer, Label>> iterator() {
        return new ImmutableInfoIterator(idLabelMap);
    }

    @Override
    public boolean domainAndIDEquals(ImmutableOutputInfo<Label> other) {
        if (size() == other.size()) {
            for (Map.Entry<Integer,String> e : idLabelMap.entrySet()) {
                Label otherLbl = other.getOutput(e.getKey());
                if (otherLbl == null) {
                    return false;
                } else if (!otherLbl.label.equals(e.getValue())) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    /**
     * An iterator that converts {@link Map.Entry} into {@link Pair}s on the way out.
     */
    private static class ImmutableInfoIterator implements Iterator<Pair<Integer,Label>> {

        private final Iterator<Map.Entry<Integer,String>> itr;

        public ImmutableInfoIterator(Map<Integer,String> idLabelMap) {
            itr = idLabelMap.entrySet().iterator();
        }

        @Override
        public boolean hasNext() {
            return itr.hasNext();
        }

        @Override
        public Pair<Integer, Label> next() {
            Map.Entry<Integer,String> e = itr.next();
            return new Pair<>(e.getKey(),new Label(e.getValue()));
        }
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        domain = Collections.unmodifiableSet(new HashSet<>(labels.values()));
    }
}
