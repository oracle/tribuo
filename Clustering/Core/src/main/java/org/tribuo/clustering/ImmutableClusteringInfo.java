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
import com.oracle.labs.mlrg.olcut.util.MutableNumber;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.clustering.protos.ClusteringInfoProto;
import org.tribuo.protos.ProtoSerializableClass;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * An {@link ImmutableOutputInfo} object for ClusterIDs.
 * <p>
 * Gives each unique cluster an id number. Also counts each id occurrence like {@link MutableClusteringInfo} does,
 * though the counts are frozen in this object.
 */
@ProtoSerializableClass(serializedDataClass = ClusteringInfoProto.class, version = 0)
public class ImmutableClusteringInfo extends ClusteringInfo implements ImmutableOutputInfo<ClusterID> {
    private static final long serialVersionUID = 1L;

    private final Set<ClusterID> domain;

    /**
     * Constructs an immutable clustering info from the supplied cluster counts.
     * @param counts The cluster counts.
     */
    public ImmutableClusteringInfo(Map<Integer,MutableLong> counts) {
        super();
        clusterCounts.putAll(MutableNumber.copyMap(counts));

        Set<ClusterID> outputs = new HashSet<>();
        for (Map.Entry<Integer,MutableLong> e : clusterCounts.entrySet()) {
            outputs.add(new ClusterID(e.getKey()));
        }
        domain = Collections.unmodifiableSet(outputs);
    }

    /**
     * Copies the supplied clustering info, generating id numbers.
     * @param other The clustering info to copy.
     */
    public ImmutableClusteringInfo(ClusteringInfo other) {
        super(other);
        Set<ClusterID> outputs = new HashSet<>();
        for (Map.Entry<Integer,MutableLong> e : clusterCounts.entrySet()) {
            outputs.add(new ClusterID(e.getKey()));
        }
        domain = Collections.unmodifiableSet(outputs);
    }

    /**
     * Deserialization constructor.
     * @param counts Counts map.
     * @param unknownCount Unknown count.
     */
    private ImmutableClusteringInfo(Map<Integer, MutableLong> counts, int unknownCount) {
        super(counts,unknownCount);
        Set<ClusterID> outputs = new HashSet<>();
        for (Map.Entry<Integer,MutableLong> e : clusterCounts.entrySet()) {
            outputs.add(new ClusterID(e.getKey()));
        }
        domain = Collections.unmodifiableSet(outputs);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static ImmutableClusteringInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
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
        return new ImmutableClusteringInfo(labelCounts,proto.getUnknownCount());
    }

    @Override
    public Set<ClusterID> getDomain() {
        return domain;
    }

    @Override
    public int getID(ClusterID output) {
        return output.getID();
    }

    @Override
    public ClusterID getOutput(int id) {
        return new ClusterID(id);
    }

    @Override
    public long getTotalObservations() {
        long count = 0;
        for (Map.Entry<Integer,MutableLong> e : clusterCounts.entrySet()) {
            count += e.getValue().longValue();
        }
        return count;
    }

    @Override
    public ClusteringInfo copy() {
        return new ImmutableClusteringInfo(this);
    }

    @Override
    public Iterator<Pair<Integer, ClusterID>> iterator() {
        return new ImmutableInfoIterator(clusterCounts.keySet());
    }

    @Override
    public boolean domainAndIDEquals(ImmutableOutputInfo<ClusterID> other) {
        return getDomain().equals(other.getDomain());
    }

    private static class ImmutableInfoIterator implements Iterator<Pair<Integer,ClusterID>> {

        private final Iterator<Integer> itr;

        public ImmutableInfoIterator(Set<Integer> idLabelMap) {
            itr = idLabelMap.iterator();
        }

        @Override
        public boolean hasNext() {
            return itr.hasNext();
        }

        @Override
        public Pair<Integer, ClusterID> next() {
            int id = itr.next();
            return new Pair<>(id, new ClusterID(id));
        }
    }
}
