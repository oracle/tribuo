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
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

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

    @Test
    public void load431Protobufs() throws URISyntaxException, IOException {
        // ClusterID
        Path clusteridPath = Paths.get(SerializationTest.class.getResource("clusterid-clustering-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(clusteridPath)) {
            OutputProto proto = OutputProto.parseFrom(fis);
            ClusterID clusterid = (ClusterID) Output.deserialize(proto);
            assertEquals(ONE, clusterid);
        }

        // ClusteringFactory
        Path factoryPath = Paths.get(SerializationTest.class.getResource("factory-clustering-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(factoryPath)) {
            OutputFactoryProto proto = OutputFactoryProto.parseFrom(fis);
            ClusteringFactory factory = (ClusteringFactory) OutputFactory.deserialize(proto);
            assertEquals(new ClusteringFactory(), factory);
        }

        MutableClusteringInfo info = new MutableClusteringInfo();
        for (int i = 0; i < 5; i++) {
            info.observe(ZERO);
            info.observe(ONE);
            info.observe(TWO);
        }
        for (int i = 0; i < 2; i++) {
            info.observe(ClusteringFactory.UNASSIGNED_CLUSTER_ID);
        }
        ImmutableClusteringInfo imInfo = (ImmutableClusteringInfo) info.generateImmutableOutputInfo();

        // MutableClusteringInfo
        Path mutablePath = Paths.get(SerializationTest.class.getResource("mutableinfo-clustering-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(mutablePath)) {
            OutputDomainProto proto = OutputDomainProto.parseFrom(fis);
            ClusteringInfo deserInfo = (ClusteringInfo) OutputInfo.deserialize(proto);
            assertEquals(info, deserInfo);
        }
        // ImmutableClusteringInfo
        Path immutablePath = Paths.get(SerializationTest.class.getResource("immutableinfo-clustering-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(immutablePath)) {
            OutputDomainProto proto = OutputDomainProto.parseFrom(fis);
            ClusteringInfo deserInfo = (ClusteringInfo) OutputInfo.deserialize(proto);
            assertEquals(imInfo, deserInfo);
        }
    }

    public void generateProtobufs() throws IOException {
        Helpers.writeProtobuf(new ClusteringFactory(), Paths.get("src","test","resources","org","tribuo","clustering","factory-clustering-431.tribuo"));
        Helpers.writeProtobuf(ONE, Paths.get("src","test","resources","org","tribuo","clustering","clusterid-clustering-431.tribuo"));
        MutableClusteringInfo info = new MutableClusteringInfo();
        for (int i = 0; i < 5; i++) {
            info.observe(ZERO);
            info.observe(ONE);
            info.observe(TWO);
        }
        for (int i = 0; i < 2; i++) {
            info.observe(ClusteringFactory.UNASSIGNED_CLUSTER_ID);
        }
        Helpers.writeProtobuf(info, Paths.get("src","test","resources","org","tribuo","clustering","mutableinfo-clustering-431.tribuo"));
        ImmutableClusteringInfo imInfo = (ImmutableClusteringInfo) info.generateImmutableOutputInfo();
        Helpers.writeProtobuf(imInfo, Paths.get("src","test","resources","org","tribuo","clustering","immutableinfo-clustering-431.tribuo"));
    }
}
