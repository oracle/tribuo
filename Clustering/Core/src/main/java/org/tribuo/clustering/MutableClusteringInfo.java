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

package org.tribuo.clustering;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.MutableOutputInfo;
import org.tribuo.clustering.protos.ClusteringInfoProto;
import org.tribuo.protos.ProtoSerializableClass;

import java.util.HashMap;
import java.util.Map;

/**
 * A mutable {@link ClusteringInfo}. Can record new observations of {@link ClusterID}s, incrementing the
 * appropriate counts.
 */
@ProtoSerializableClass(serializedDataClass = ClusteringInfoProto.class, version = 0)
public class MutableClusteringInfo extends ClusteringInfo implements MutableOutputInfo<ClusterID> {
    private static final long serialVersionUID = 1L;

    MutableClusteringInfo() {
        super();
    }

    MutableClusteringInfo(ClusteringInfo info) {
        super(info);
    }

    /**
     * Deserialization constructor.
     * @param counts Counts map.
     * @param unknownCount Unknown count.
     */
    private MutableClusteringInfo(Map<Integer, MutableLong> counts, int unknownCount) {
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
    public static MutableClusteringInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        ClusteringInfoProto proto = message.unpack(ClusteringInfoProto.class);
        if (proto.getIdCount() != proto.getCountCount()) {
            throw new IllegalArgumentException("Invalid protobuf, different numbers of ids and counts, labels " + proto.getIdCount() + ", counts " + proto.getCountCount());
        }
        Map<Integer,MutableLong> labelCounts = new HashMap<>();
        for (int i = 0; i < proto.getIdCount(); i++) {
            Integer lbl = proto.getId(i);
            long cnt = proto.getCount(i);
            MutableLong old = labelCounts.put(lbl,new MutableLong(cnt));
            if (old != null) {
                throw new IllegalArgumentException("Invalid protobuf, two mappings for " + lbl);
            }
        }
        return new MutableClusteringInfo(labelCounts,proto.getUnknownCount());
    }

    @Override
    public void observe(ClusterID output) {
        if (output == ClusteringFactory.UNASSIGNED_CLUSTER_ID) {
            unknownCount++;
        } else {
            int id = output.getID();
            MutableLong value = clusterCounts.computeIfAbsent(id, k -> new MutableLong());
            value.increment();
        }
    }

    @Override
    public void clear() {
        clusterCounts.clear();
    }

    @Override
    public MutableClusteringInfo copy() {
        return new MutableClusteringInfo(this);
    }

}
