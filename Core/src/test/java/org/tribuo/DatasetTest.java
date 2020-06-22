/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo;

import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.tribuo.test.Helpers.mkExample;

/**
 **/
public class DatasetTest {

    /**
     * Tests that list returned from {@link Dataset#getData()} is unmodifiable.
     */
    @Test
    public void testGetData() {
        OutputFactory<MockOutput> outputFactory = new MockOutputFactory();
        MutableDataset<MockOutput> a = new MutableDataset<>(new MockDataSourceProvenance(), outputFactory);
        Assertions.assertThrows(UnsupportedOperationException.class, () -> a.getData().add(mkExample(outputFactory.generateOutput("1"), "a")), "Expected exception thrown as adding to unmodifiable list.");
    }
}