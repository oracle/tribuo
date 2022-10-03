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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.MutableOutputInfo;
import org.tribuo.classification.protos.MutableLabelInfoProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.OutputDomainProto;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A mutable {@link LabelInfo}. Can record new observations of Labels, incrementing the
 * appropriate counts.
 */
@ProtoSerializableClass(serializedDataClass=MutableLabelInfoProto.class, version=0)
public class MutableLabelInfo extends LabelInfo implements MutableOutputInfo<Label> {
    private static final long serialVersionUID = 1L;

    MutableLabelInfo() {
        super();
    }

    /**
     * Constructs a mutable deep copy of the supplied label info.
     * @param info The info to copy.
     */
    public MutableLabelInfo(LabelInfo info) {
        super(info);
    }

    /**
     * Deserialization constructor.
     * @param counts Counts map.
     * @param unknownCount Unknown count.
     */
    private MutableLabelInfo(Map<String, MutableLong> counts, int unknownCount) {
        super(counts,unknownCount);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static MutableLabelInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        MutableLabelInfoProto proto = message.unpack(MutableLabelInfoProto.class);
        if (proto.getLabelCount() != proto.getCountCount()) {
            throw new IllegalArgumentException("Invalid protobuf, different numbers of labels and counts, labels " + proto.getLabelCount() + ", counts " + proto.getCountCount());
        }
        Map<String,MutableLong> labelCounts = new HashMap<>();
        for (int i = 0; i < proto.getLabelCount(); i++) {
            String lbl = proto.getLabel(i);
            long cnt = proto.getCount(i);
            MutableLong old = labelCounts.put(lbl,new MutableLong(cnt));
            if (old != null) {
                throw new IllegalArgumentException("Invalid protobuf, two mappings for " + lbl);
            }
        }
        return new MutableLabelInfo(labelCounts,proto.getUnknownCount());
    }

    @Override
    public OutputDomainProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public void observe(Label output) {
        if (output == LabelFactory.UNKNOWN_LABEL) {
            unknownCount++;
        } else {
            String label = output.getLabel();
            MutableLong value = labelCounts.computeIfAbsent(label, k -> new MutableLong());
            labels.computeIfAbsent(label, Label::new);
            value.increment();
        }
    }

    @Override
    public void clear() {
        labelCounts.clear();
    }

    @Override
    public MutableLabelInfo copy() {
        return new MutableLabelInfo(this);
    }

    @Override
    public String toReadableString() {
        StringBuilder builder = new StringBuilder();
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            if (builder.length() > 0) {
                builder.append(", ");
            }
            builder.append('(');
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
        MutableLabelInfo labelInfo = (MutableLabelInfo) o;
        if (unknownCount == labelInfo.unknownCount && labelCounts.size() == labelInfo.labelCounts.size()) {
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
        return Objects.hash(labelCounts, unknownCount);
    }
}
