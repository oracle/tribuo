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

package org.tribuo.regression;

import org.junit.jupiter.api.Test;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.protos.core.OutputDomainProto;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.protos.core.OutputProto;

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

}
