/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

import org.junit.jupiter.api.Test;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.protos.core.OutputDomainProto;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.protos.core.OutputProto;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SerializationTest {

    private static final ClusterID ZERO = new ClusterID(0);
    private static final ClusterID ONE = new ClusterID(1);
    private static final ClusterID TWO = new ClusterID(2);

    @Test
    public void clusterSerializationTest() {
        OutputProto oneSer = ONE.serialize();
        ClusterID oneDeser = (ClusterID) Output.deserialize(oneSer);
        assertEquals(ONE,oneDeser);
    }

    @Test
    public void factorySerializationTest() {
        OutputFactory<ClusterID> clusterFactory = new ClusteringFactory();
        OutputFactoryProto factorySer = clusterFactory.serialize();
        OutputFactory<?> factoryDeser = OutputFactory.deserialize(factorySer);
        assertEquals(clusterFactory,factoryDeser);
    }

    @Test
    public void infoSerializationTest() {
        MutableClusteringInfo info = new MutableClusteringInfo();

        for (int i = 0; i < 5; i++) {
            info.observe(ZERO);
            info.observe(ONE);
            info.observe(TWO);
        }

        for (int i = 0; i < 2; i++) {
            info.observe(ClusteringFactory.UNASSIGNED_CLUSTER_ID);
        }

        OutputDomainProto serInfo = info.serialize();
        MutableClusteringInfo deserInfo = (MutableClusteringInfo) OutputInfo.deserialize(serInfo);

        assertEquals(info,deserInfo);

        ImmutableClusteringInfo immutableInfo = new ImmutableClusteringInfo(info);
        OutputDomainProto serImInfo = immutableInfo.serialize();
        ImmutableClusteringInfo deserImInfo = (ImmutableClusteringInfo) OutputInfo.deserialize(serImInfo);
        assertEquals(immutableInfo,deserImInfo);
    }

}
