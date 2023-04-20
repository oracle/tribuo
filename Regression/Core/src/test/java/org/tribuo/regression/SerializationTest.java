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

package org.tribuo.regression;

import org.junit.jupiter.api.Test;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.protos.core.EnsembleCombinerProto;
import org.tribuo.protos.core.OutputDomainProto;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.protos.core.OutputProto;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SerializationTest {

    private static final Regressor.DimensionTuple A = new Regressor.DimensionTuple("ONE",1.0);
    private static final Regressor.DimensionTuple B = new Regressor.DimensionTuple("TWO",2.0);
    private static final Regressor.DimensionTuple C = new Regressor.DimensionTuple("ONE",0.5);
    private static final Regressor.DimensionTuple D = new Regressor.DimensionTuple("TWO",5.0);

    private static final Regressor ONE = new Regressor(new Regressor.DimensionTuple[]{A,B});
    private static final Regressor TWO = new Regressor(new Regressor.DimensionTuple[]{C,D});

    @Test
    public void regressorSerializationTest() {
        OutputProto oneSer = ONE.serialize();
        Regressor oneDeser = (Regressor) Output.deserialize(oneSer);
        assertEquals(ONE,oneDeser);
        OutputProto dimSer = A.serialize();
        Regressor dimDeser = (Regressor) Output.deserialize(dimSer);
        assertEquals(A, dimDeser);
    }

    @Test
    public void factorySerializationTest() {
        OutputFactory<Regressor> lblFactory = new RegressionFactory();
        OutputFactoryProto factorySer = lblFactory.serialize();
        OutputFactory<?> factoryDeser = OutputFactory.deserialize(factorySer);
        assertEquals(lblFactory,factoryDeser);
    }

    @Test
    public void infoSerializationTest() {
        MutableRegressionInfo info = new MutableRegressionInfo();

        for (int i = 0; i < 5; i++) {
            info.observe(ONE);
            info.observe(TWO);
        }

        for (int i = 0; i < 2; i++) {
            info.observe(RegressionFactory.UNKNOWN_REGRESSOR);
        }

        OutputDomainProto serInfo = info.serialize();
        MutableRegressionInfo deserInfo = (MutableRegressionInfo) OutputInfo.deserialize(serInfo);

        assertEquals(info,deserInfo);

        ImmutableRegressionInfo immutableInfo = new ImmutableRegressionInfo(info);
        OutputDomainProto serImInfo = immutableInfo.serialize();
        ImmutableRegressionInfo deserImInfo = (ImmutableRegressionInfo) OutputInfo.deserialize(serImInfo);
        assertEquals(immutableInfo,deserImInfo);
    }

    @Test
    public void load431Protobufs() throws URISyntaxException, IOException {
        // Regressor
        Path regressorPath = Paths.get(SerializationTest.class.getResource("regressor-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(regressorPath)) {
            OutputProto proto = OutputProto.parseFrom(fis);
            Regressor regression = (Regressor) Output.deserialize(proto);
            assertEquals(ONE, regression);
        }
        // Regressor.DimensionTuple
        Path dimPath = Paths.get(SerializationTest.class.getResource("dimension-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(dimPath)) {
            OutputProto proto = OutputProto.parseFrom(fis);
            Regressor regression = (Regressor) Output.deserialize(proto);
            assertEquals(A, regression);
        }

        // RegressionFactory
        Path factoryPath = Paths.get(SerializationTest.class.getResource("factory-regression-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(factoryPath)) {
            OutputFactoryProto proto = OutputFactoryProto.parseFrom(fis);
            RegressionFactory factory = (RegressionFactory) OutputFactory.deserialize(proto);
            assertEquals(new RegressionFactory(), factory);
        }

        MutableRegressionInfo info = new MutableRegressionInfo();
        for (int i = 0; i < 5; i++) {
            info.observe(ONE);
            info.observe(TWO);
        }
        for (int i = 0; i < 2; i++) {
            info.observe(RegressionFactory.UNKNOWN_REGRESSOR);
        }
        ImmutableRegressionInfo imInfo = (ImmutableRegressionInfo) info.generateImmutableOutputInfo();

        // MutableRegressionInfo
        Path mutablePath = Paths.get(SerializationTest.class.getResource("mutableinfo-regression-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(mutablePath)) {
            OutputDomainProto proto = OutputDomainProto.parseFrom(fis);
            RegressionInfo deserInfo = (RegressionInfo) OutputInfo.deserialize(proto);
            assertEquals(info, deserInfo);
        }
        // ImmutableRegressionInfo
        Path immutablePath = Paths.get(SerializationTest.class.getResource("immutableinfo-regression-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(immutablePath)) {
            OutputDomainProto proto = OutputDomainProto.parseFrom(fis);
            RegressionInfo deserInfo = (RegressionInfo) OutputInfo.deserialize(proto);
            assertEquals(imInfo, deserInfo);
        }
        // AveragingCombiner
        AveragingCombiner comb = new AveragingCombiner();
        Path combinerPath = Paths.get(SerializationTest.class.getResource("combiner-regression-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(combinerPath)) {
            EnsembleCombinerProto proto = EnsembleCombinerProto.parseFrom(fis);
            AveragingCombiner deserComb = (AveragingCombiner) EnsembleCombiner.deserialize(proto);
            assertEquals(comb, deserComb);
        }
    }

    public void generateProtobufs() throws IOException {
        Helpers.writeProtobuf(new RegressionFactory(), Paths.get("src","test","resources","org","tribuo","regression","factory-regression-431.tribuo"));
        Helpers.writeProtobuf(ONE, Paths.get("src","test","resources","org","tribuo","regression","regressor-431.tribuo"));
        Helpers.writeProtobuf(A, Paths.get("src","test","resources","org","tribuo","regression","dimension-431.tribuo"));
        MutableRegressionInfo info = new MutableRegressionInfo();
        for (int i = 0; i < 5; i++) {
            info.observe(ONE);
            info.observe(TWO);
        }
        for (int i = 0; i < 2; i++) {
            info.observe(RegressionFactory.UNKNOWN_REGRESSOR);
        }
        Helpers.writeProtobuf(info, Paths.get("src","test","resources","org","tribuo","regression","mutableinfo-regression-431.tribuo"));
        ImmutableRegressionInfo imInfo = (ImmutableRegressionInfo) info.generateImmutableOutputInfo();
        Helpers.writeProtobuf(imInfo, Paths.get("src","test","resources","org","tribuo","regression","immutableinfo-regression-431.tribuo"));
        Helpers.writeProtobuf(new AveragingCombiner(), Paths.get("src","test","resources","org","tribuo","regression","combiner-regression-431.tribuo"));
    }

}
