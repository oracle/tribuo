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

package org.tribuo.classification;

import org.junit.jupiter.api.Test;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.protos.core.OutputDomainProto;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.protos.core.OutputProto;

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

}
