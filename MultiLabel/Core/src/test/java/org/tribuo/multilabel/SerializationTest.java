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

package org.tribuo.multilabel;

import org.junit.jupiter.api.Test;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.classification.Label;
import org.tribuo.protos.core.OutputDomainProto;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.protos.core.OutputProto;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SerializationTest {

    private static final Label A = new Label("A");
    private static final Label B = new Label("B");
    private static final Label C = new Label("C");
    private static final Set<Label> SET = new HashSet<>(Arrays.asList(A,B,C));
    private static final MultiLabel ONE = new MultiLabel("ONE");
    private static final MultiLabel TWO = new MultiLabel(SET);

    @Test
    public void multilabelSerializationTest() {
        OutputProto oneSer = ONE.serialize();
        MultiLabel oneDeser = (MultiLabel) Output.deserialize(oneSer);
        assertEquals(ONE,oneDeser);
    }

    @Test
    public void factorySerializationTest() {
        OutputFactory<MultiLabel> lblFactory = new MultiLabelFactory();
        OutputFactoryProto factorySer = lblFactory.serialize();
        OutputFactory<?> factoryDeser = OutputFactory.deserialize(factorySer);
        assertEquals(lblFactory,factoryDeser);
    }

    @Test
    public void infoSerializationTest() {
        MutableMultiLabelInfo info = new MutableMultiLabelInfo();

        for (int i = 0; i < 5; i++) {
            info.observe(ONE);
            info.observe(TWO);
        }

        for (int i = 0; i < 2; i++) {
            info.observe(MultiLabelFactory.UNKNOWN_MULTILABEL);
        }

        OutputDomainProto serInfo = info.serialize();
        MutableMultiLabelInfo deserInfo = (MutableMultiLabelInfo) OutputInfo.deserialize(serInfo);

        assertEquals(info,deserInfo);

        ImmutableMultiLabelInfo immutableInfo = new ImmutableMultiLabelInfo(info);
        OutputDomainProto serImInfo = immutableInfo.serialize();
        ImmutableMultiLabelInfo deserImInfo = (ImmutableMultiLabelInfo) OutputInfo.deserialize(serImInfo);
        assertEquals(immutableInfo,deserImInfo);
    }

}
