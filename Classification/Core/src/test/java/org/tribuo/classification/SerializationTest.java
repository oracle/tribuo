/*
 * Copyright (c) 2022, 2023, Oracle and/or its affiliates. All rights reserved.
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

import org.junit.jupiter.api.Test;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.classification.ensemble.FullyWeightedVotingCombiner;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.protos.core.EnsembleCombinerProto;
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

    private static final Label ONE = new Label("ONE");
    private static final Label TWO = new Label("TWO");

    @Test
    public void labelSerializationTest() {
        OutputProto oneSer = ONE.serialize();
        Label oneDeser = (Label) Output.deserialize(oneSer);
        assertEquals(ONE,oneDeser);
    }

    @Test
    public void factorySerializationTest() {
        OutputFactory<Label> lblFactory = new LabelFactory();
        OutputFactoryProto factorySer = lblFactory.serialize();
        OutputFactory<?> factoryDeser = OutputFactory.deserialize(factorySer);
        assertEquals(lblFactory,factoryDeser);
    }

    @Test
    public void infoSerializationTest() {
        MutableLabelInfo info = new MutableLabelInfo();

        for (int i = 0; i < 5; i++) {
            info.observe(ONE);
            info.observe(TWO);
        }

        for (int i = 0; i < 2; i++) {
            info.observe(LabelFactory.UNKNOWN_LABEL);
        }

        OutputDomainProto serInfo = info.serialize();
        MutableLabelInfo deserInfo = (MutableLabelInfo) OutputInfo.deserialize(serInfo);

        assertEquals(info,deserInfo);

        ImmutableLabelInfo immutableInfo = new ImmutableLabelInfo(info);
        OutputDomainProto serImInfo = immutableInfo.serialize();
        ImmutableLabelInfo deserImInfo = (ImmutableLabelInfo) OutputInfo.deserialize(serImInfo);
        assertEquals(immutableInfo,deserImInfo);
    }

    @Test
    public void load431Protobufs() throws URISyntaxException, IOException {
        Label test = new Label("TEST",1.0);
        Label other = new Label("OTHER",1.0);
        // Label
        Path eventPath = Paths.get(SerializationTest.class.getResource("label-clf-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(eventPath)) {
            OutputProto proto = OutputProto.parseFrom(fis);
            Label lbl = (Label) Output.deserialize(proto);
            assertEquals(test, lbl);
        }

        // LabelFactory
        Path factoryPath = Paths.get(SerializationTest.class.getResource("factory-clf-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(factoryPath)) {
            OutputFactoryProto proto = OutputFactoryProto.parseFrom(fis);
            LabelFactory factory = (LabelFactory) OutputFactory.deserialize(proto);
            assertEquals(new LabelFactory(), factory);
        }

        MutableLabelInfo info = new MutableLabelInfo();
        for (int i = 0; i < 5; i++) {
            info.observe(test);
            info.observe(other);
        }
        for (int i = 0; i < 2; i++) {
            info.observe(LabelFactory.UNKNOWN_LABEL);
        }
        ImmutableLabelInfo imInfo = (ImmutableLabelInfo) info.generateImmutableOutputInfo();

        // MutableLabelInfo
        Path mutablePath = Paths.get(SerializationTest.class.getResource("mutableinfo-clf-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(mutablePath)) {
            OutputDomainProto proto = OutputDomainProto.parseFrom(fis);
            LabelInfo deserInfo = (LabelInfo) OutputInfo.deserialize(proto);
            assertEquals(info, deserInfo);
        }
        // ImmutableLabelInfo
        Path immutablePath = Paths.get(SerializationTest.class.getResource("immutableinfo-clf-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(immutablePath)) {
            OutputDomainProto proto = OutputDomainProto.parseFrom(fis);
            LabelInfo deserInfo = (LabelInfo) OutputInfo.deserialize(proto);
            assertEquals(imInfo, deserInfo);
        }
        // VotingCombiner
        VotingCombiner comb = new VotingCombiner();
        Path combinerPath = Paths.get(SerializationTest.class.getResource("vote-combiner-clf-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(combinerPath)) {
            EnsembleCombinerProto proto = EnsembleCombinerProto.parseFrom(fis);
            VotingCombiner deserComb = (VotingCombiner) EnsembleCombiner.deserialize(proto);
            assertEquals(comb, deserComb);
        }
        // MultiLabelVotingCombiner
        FullyWeightedVotingCombiner fvComb = new FullyWeightedVotingCombiner();
        Path fvCombinerPath = Paths.get(SerializationTest.class.getResource("fullvote-combiner-clf-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(fvCombinerPath)) {
            EnsembleCombinerProto proto = EnsembleCombinerProto.parseFrom(fis);
            FullyWeightedVotingCombiner deserComb = (FullyWeightedVotingCombiner) EnsembleCombiner.deserialize(proto);
            assertEquals(fvComb, deserComb);
        }
    }

    public void generateProtobufs() throws IOException {
        Label test = new Label("TEST",1.0);
        Label other = new Label("OTHER",1.0);
        Helpers.writeProtobuf(new LabelFactory(), Paths.get("src","test","resources","org","tribuo","classification","factory-clf-431.tribuo"));
        Helpers.writeProtobuf(test, Paths.get("src","test","resources","org","tribuo","classification","label-clf-431.tribuo"));
        MutableLabelInfo info = new MutableLabelInfo();
        for (int i = 0; i < 5; i++) {
            info.observe(test);
            info.observe(other);
        }
        for (int i = 0; i < 2; i++) {
            info.observe(LabelFactory.UNKNOWN_LABEL);
        }
        Helpers.writeProtobuf(info, Paths.get("src","test","resources","org","tribuo","classification","mutableinfo-clf-431.tribuo"));
        ImmutableLabelInfo imInfo = (ImmutableLabelInfo) info.generateImmutableOutputInfo();
        Helpers.writeProtobuf(imInfo, Paths.get("src","test","resources","org","tribuo","classification","immutableinfo-clf-431.tribuo"));
        Helpers.writeProtobuf(new VotingCombiner(), Paths.get("src","test","resources","org","tribuo","classification","vote-combiner-clf-431.tribuo"));
        Helpers.writeProtobuf(new FullyWeightedVotingCombiner(), Paths.get("src","test","resources","org","tribuo","classification","fullvote-combiner-clf-431.tribuo"));
    }
}
