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

package org.tribuo.classification.sequence.viterbi;

import org.junit.jupiter.api.Test;
import org.tribuo.classification.Label;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ViterbiModelTest {

    @Test
    public void testGetTopOutcomes() {

        Map<String, Label> scoredOutcomes = new HashMap<>();
        scoredOutcomes.put("A", new Label("A", 0.1d));
        scoredOutcomes.put("B", new Label("B", 0.2d));
        scoredOutcomes.put("C", new Label("C", 0.15d));
        scoredOutcomes.put("D", new Label("D", 0.25d));

        List<Label> topOutcomes = ViterbiModel.getTopLabels(scoredOutcomes, 4);
        assertEquals(4, topOutcomes.size());
        assertEquals("D", topOutcomes.get(0).getLabel());
        assertEquals("B", topOutcomes.get(1).getLabel());
        assertEquals("C", topOutcomes.get(2).getLabel());
        assertEquals("A", topOutcomes.get(3).getLabel());
    }

}
