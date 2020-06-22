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

package org.tribuo.clustering;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.MutableOutputInfo;

/**
 * A mutable {@link ClusteringInfo}. Can record new observations of {@link ClusterID}s, incrementing the
 * appropriate counts.
 */
public class MutableClusteringInfo extends ClusteringInfo implements MutableOutputInfo<ClusterID> {
    private static final long serialVersionUID = 1L;

    MutableClusteringInfo() {
        super();
    }

    MutableClusteringInfo(ClusteringInfo info) {
        super(info);
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
