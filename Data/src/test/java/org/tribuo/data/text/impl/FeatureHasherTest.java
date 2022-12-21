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

package org.tribuo.data.text.impl;

import org.junit.jupiter.api.Test;
import org.tribuo.Feature;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class FeatureHasherTest {

    @Test
    public void negativeValuesTest() {
        List<Feature> input = new ArrayList<>();
        Feature posValue = new Feature("Testing", 2.0);
        input.add(posValue);
        Feature negValue = new Feature("Test",2.0);
        input.add(negValue);

        FeatureHasher preserving = new FeatureHasher(10, true);
        FeatureHasher notPreserving = new FeatureHasher(10, false);

        List<Feature> preservingOutput = preserving.map("test", input);
        List<Feature> notPreservingOutput = notPreserving.map("test", input);

        assertEquals(2.0, preservingOutput.get(0).getValue());
        assertEquals(2.0, preservingOutput.get(1).getValue());

        assertEquals(1.0, notPreservingOutput.get(0).getValue());
        assertEquals(-1.0, notPreservingOutput.get(1).getValue());
    }

}
