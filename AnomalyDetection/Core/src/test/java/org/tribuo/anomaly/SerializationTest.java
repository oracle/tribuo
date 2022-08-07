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

package org.tribuo.anomaly;

import org.junit.jupiter.api.Test;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.protos.core.OutputDomainProto;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.protos.core.OutputProto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.tribuo.anomaly.AnomalyFactory.ANOMALOUS_EVENT;
import static org.tribuo.anomaly.AnomalyFactory.EXPECTED_EVENT;
import static org.tribuo.anomaly.AnomalyFactory.UNKNOWN_EVENT;

public class SerializationTest {

    @Test
    public void eventSerializationTest() {
        OutputProto oneSer = ANOMALOUS_EVENT.serialize();
        Event oneDeser = (Event) Output.deserialize(oneSer);
        assertEquals(ANOMALOUS_EVENT,oneDeser);
    }

    @Test
    public void factorySerializationTest() {
        OutputFactory<Event> anomalyFactory = new AnomalyFactory();
        OutputFactoryProto factorySer = anomalyFactory.serialize();
        OutputFactory<?> factoryDeser = OutputFactory.deserialize(factorySer);
        assertEquals(anomalyFactory,factoryDeser);
    }

    @Test
    public void infoSerializationTest() {
        MutableAnomalyInfo info = new MutableAnomalyInfo();

        for (int i = 0; i < 5; i++) {
            info.observe(EXPECTED_EVENT);
            info.observe(ANOMALOUS_EVENT);
        }

        for (int i = 0; i < 2; i++) {
            info.observe(UNKNOWN_EVENT);
        }

        OutputDomainProto serInfo = info.serialize();
        MutableAnomalyInfo deserInfo = (MutableAnomalyInfo) OutputInfo.deserialize(serInfo);

        assertEquals(info,deserInfo);

        ImmutableAnomalyInfo immutableInfo = new ImmutableAnomalyInfo(info);
        OutputDomainProto serImInfo = immutableInfo.serialize();
        ImmutableAnomalyInfo deserImInfo = (ImmutableAnomalyInfo) OutputInfo.deserialize(serImInfo);
        assertEquals(immutableInfo,deserImInfo);
    }

}
