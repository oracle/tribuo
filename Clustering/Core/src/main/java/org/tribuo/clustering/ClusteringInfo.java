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

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.MutableNumber;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputInfo;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoSerializableKeysValuesField;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.OutputDomainProto;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The base class for a ClusterID OutputInfo.
 */
public abstract class ClusteringInfo implements OutputInfo<ClusterID> {
    private static final long serialVersionUID = 1L;

    /**
     * The cluster counts for each id.
     */
    @ProtoSerializableKeysValuesField(keysName = "id", valuesName = "count")
    protected final Map<Integer,MutableLong> clusterCounts;
    /**
     * The number of times this object has seen the unknown cluster id.
     */
    @ProtoSerializableField
    protected int unknownCount = 0;

    ClusteringInfo() {
        clusterCounts = new HashMap<>();
    }

    ClusteringInfo(ClusteringInfo other) {
        clusterCounts = MutableNumber.copyMap(other.clusterCounts);
    }

    ClusteringInfo(Map<Integer,MutableLong> clusterCounts, int unknownCount) {
        if (unknownCount < 0) {
            throw new IllegalStateException("unknownCount must be non-negative, found " + unknownCount);
        }
        this.unknownCount = unknownCount;
        this.clusterCounts = new HashMap<>();
        for (Map.Entry<Integer,MutableLong> e : clusterCounts.entrySet()) {
            if (e.getValue().longValue() < 1) {
                throw new IllegalArgumentException("Count for " + e.getKey() + " must be positive but found " + e.getValue().longValue());
            }
            if (e.getKey() < 0) {
                throw new IllegalArgumentException("Id " + e.getKey() + " is invalid, all ids must be non-negative");
            }
            this.clusterCounts.put(e.getKey(),e.getValue());
        }
    }

    @Override
    public OutputDomainProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public int getUnknownCount() {
        return unknownCount;
    }

    @Override
    public Set<ClusterID> getDomain() {
        Set<ClusterID> outputs = new HashSet<>();
        for (Map.Entry<Integer,MutableLong> e : clusterCounts.entrySet()) {
            outputs.add(new ClusterID(e.getKey()));
        }
        return outputs;
    }

    @Override
    public int size() {
        return clusterCounts.size();
    }

    @Override
    public ImmutableOutputInfo<ClusterID> generateImmutableOutputInfo() {
        return new ImmutableClusteringInfo(this);
    }

    @Override
    public MutableOutputInfo<ClusterID> generateMutableOutputInfo() {
        return new MutableClusteringInfo(this);
    }

    @Override
    public abstract ClusteringInfo copy();

    @Override
    public Iterable<Pair<String, Long>> outputCountsIterable() {
        return () -> new Iterator<Pair<String,Long>>() {
            Iterator<Map.Entry<Integer,MutableLong>> itr = clusterCounts.entrySet().iterator();

            @Override
            public boolean hasNext() {
                return itr.hasNext();
            }

            @Override
            public Pair<String,Long> next() {
                Map.Entry<Integer,MutableLong> e = itr.next();
                return new Pair<>(""+e.getKey(),e.getValue().longValue());
            }
        };
    }

    @Override
    public String toReadableString() {
        StringBuilder builder = new StringBuilder();
        for (Map.Entry<Integer,MutableLong> e : clusterCounts.entrySet()) {
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
        if (o == null || !(o instanceof ClusteringInfo)) return false;
        ClusteringInfo clusterInfo = (ClusteringInfo) o;
        if (unknownCount == clusterInfo.unknownCount && clusterCounts.size() == clusterInfo.clusterCounts.size()) {
            for (Map.Entry<Integer,MutableLong> e : clusterCounts.entrySet()) {
                MutableLong other = clusterInfo.clusterCounts.get(e.getKey());
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
        return Objects.hash(clusterCounts, unknownCount);
    }
}
