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

package org.tribuo.clustering;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.MutableNumber;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;

import java.util.Collections;
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
