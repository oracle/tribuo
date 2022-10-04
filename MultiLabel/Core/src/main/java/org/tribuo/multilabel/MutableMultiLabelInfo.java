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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.MutableOutputInfo;
import org.tribuo.multilabel.protos.MutableMultiLabelInfoProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.OutputDomainProto;

import java.util.HashMap;
import java.util.Map;

/**
 * A MutableOutputInfo for working with multi-label tasks.
 */
@ProtoSerializableClass(serializedDataClass=MutableMultiLabelInfoProto.class, version=0)
public class MutableMultiLabelInfo extends MultiLabelInfo implements MutableOutputInfo<MultiLabel> {
    private static final long serialVersionUID = 1L;

    /**
     * Package private constructor for building MutableMultiLabelInfo, used by {@link MultiLabelFactory}.
     */
    MutableMultiLabelInfo() {
        super();
    }

    /**
     * Construct a MutableMultiLabelInfo with it's state copied from another
     * MultiLabelInfo.
     * @param info The info to copy.
     */
    public MutableMultiLabelInfo(MultiLabelInfo info) {
        super(info);
    }

    /**
     * Deserialization constructor.
     * @param counts Counts map.
     * @param unknownCount Unknown count.
     * @param totalCount Total count.
     */
    private MutableMultiLabelInfo(Map<String, MutableLong> counts, int unknownCount, int totalCount) {
        super(counts,unknownCount,totalCount);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static MutableMultiLabelInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        MutableMultiLabelInfoProto proto = message.unpack(MutableMultiLabelInfoProto.class);
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
        return new MutableMultiLabelInfo(labelCounts,proto.getUnknownCount(),proto.getTotalCount());
    }

    @Override
    public OutputDomainProto serialize() {
        return ProtoUtil.serialize(this);
    }

    /**
     * Throws IllegalStateException if the MultiLabel contains a Label which has a "," in it.
     * <p>
     * Such labels are disallowed. There should be an exception thrown when one is constructed
     * too.
     * @param output The observed output.
     */
    @Override
    public void observe(MultiLabel output) {
        if (output == MultiLabelFactory.UNKNOWN_MULTILABEL) {
            unknownCount++;
        } else {
            for (String label : output.getNameSet()) {
                if (label.contains(",")) {
                    throw new IllegalStateException("MultiLabel cannot use a Label which contains ','. The supplied label was " + label + ".");
                }
                MutableLong value = labelCounts.computeIfAbsent(label, k -> new MutableLong());
                labels.computeIfAbsent(label, MultiLabel::new);
                value.increment();
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
    public MutableMultiLabelInfo copy() {
        return new MutableMultiLabelInfo(this);
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
}
