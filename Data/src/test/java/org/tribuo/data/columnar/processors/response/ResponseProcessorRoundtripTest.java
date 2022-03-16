/*
 * Copyright (c) 2020-2022 Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.data.columnar.processors.response;

import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockMultiOutput;
import org.tribuo.test.MockMultiOutputFactory;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.util.Arrays;

public class ResponseProcessorRoundtripTest {

    @Test
    public void binaryTest() {
        BinaryResponseProcessor<MockMultiOutput> multiRespProc = new BinaryResponseProcessor<>(
                Arrays.asList("R1", "R2"),
                Arrays.asList("TRUE", "TRUE"),
                new MockMultiOutputFactory(),
                "true", "false", true);

        Helpers.testConfigurableRoundtrip(multiRespProc);

        BinaryResponseProcessor<MockOutput> singleRespProc = new BinaryResponseProcessor<>("R1", "TRUE", new MockOutputFactory());

        Helpers.testConfigurableRoundtrip(singleRespProc);
    }

    @Test
    public void fieldTest() {
        FieldResponseProcessor<MockMultiOutput> multiRespProc = new FieldResponseProcessor<>(
                Arrays.asList("R1", "R2"),
                Arrays.asList("A", "B"),
                new MockMultiOutputFactory(),
                true, false);

        Helpers.testConfigurableRoundtrip(multiRespProc);

        FieldResponseProcessor<MockOutput> singleRespProc = new FieldResponseProcessor<>("R1", "A", new MockOutputFactory());

        Helpers.testConfigurableRoundtrip(singleRespProc);
    }
}
