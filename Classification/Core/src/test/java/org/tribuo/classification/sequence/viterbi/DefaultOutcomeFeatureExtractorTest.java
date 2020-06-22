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
import org.tribuo.Feature;
import org.tribuo.classification.Label;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DefaultOutcomeFeatureExtractorTest {

    @Test
    public void testExtractFeatures() {
        LabelFeatureExtractor ofe = new DefaultFeatureExtractor();
        double one = 1.0;

        List<Feature> features = ofe.extractFeatures(Arrays.asList(new Label("A"), new Label("B"), new Label("C"), new Label("D")), one);
        assertEquals(5, features.size());
        assertEquals("PreviousOutcome_L1_D", features.get(0).getName());
        assertEquals("PreviousOutcome_L2_C", features.get(1).getName());
        assertEquals("PreviousOutcome_L3_B", features.get(2).getName());
        assertEquals("PreviousOutcomes_L1_2gram_L2R_D_C", features.get(3).getName());
        assertEquals("PreviousOutcomes_L1_3gram_L2R_D_C_B", features.get(4).getName());

        features = ofe.extractFeatures(Arrays.asList(new Label("A"), new Label("B")), one);
        assertEquals(3, features.size());
        assertEquals("PreviousOutcome_L1_B", features.get(0).getName());
        assertEquals("PreviousOutcome_L2_A", features.get(1).getName());
        assertEquals("PreviousOutcomes_L1_2gram_L2R_B_A", features.get(2).getName());

        features = ofe.extractFeatures(Collections.emptyList(), one);
        assertEquals(0, features.size());

        ofe = new DefaultFeatureExtractor(2, 3, false, true, true);
        features = ofe.extractFeatures(Arrays.asList(new Label("A"), new Label("B"), new Label("C"), new Label("D")), one);
        assertEquals(4, features.size());
        assertEquals("PreviousOutcome_L2_C", features.get(0).getName());
        assertEquals("PreviousOutcome_L3_B", features.get(1).getName());
        assertEquals("PreviousOutcomes_L1_3gram_L2R_D_C_B", features.get(2).getName());
        assertEquals("PreviousOutcomes_L1_4gram_L2R_D_C_B_A", features.get(3).getName());

    }

}

