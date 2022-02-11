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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

public class EmptyResponseProcessorTest {

    @SuppressWarnings("deprecation") // due to setFieldName test
    @Test
    public void basicTest() {
        MockOutputFactory outputFactory = new MockOutputFactory();
        EmptyResponseProcessor<MockOutput> rp = new EmptyResponseProcessor<>(outputFactory);

        // Check the output factory is stored correctly
        Assertions.assertEquals(outputFactory,rp.getOutputFactory());

        // Check the field name is right
        Assertions.assertEquals(EmptyResponseProcessor.FIELD_NAME, rp.getFieldName());

        // setFieldName is a no-op on this response processor
        rp.setFieldName("Something");
        Assertions.assertEquals(EmptyResponseProcessor.FIELD_NAME, rp.getFieldName());

        // Check that it doesn't throw exceptions when given odd text, and that it always returns Optional.empty.
        Assertions.assertFalse(rp.process("").isPresent());
        Assertions.assertFalse(rp.process("test").isPresent());
        Assertions.assertFalse(rp.process("!@$#$!").isPresent());
        Assertions.assertFalse(rp.process("\n").isPresent());
        Assertions.assertFalse(rp.process("\t").isPresent());
        Assertions.assertFalse(rp.process((String) null).isPresent());
    }

}
