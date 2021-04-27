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

package org.tribuo.classification.evaluation;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.tribuo.classification.Utils.label;
import static org.tribuo.classification.Utils.mkDomain;
import static org.tribuo.classification.Utils.mkPrediction;


public class LabelConfusionMatrixTest {

    @Test
    public void testMulticlass() {
        List<Prediction<Label>> predictions = Arrays.asList(
                mkPrediction("a", "a"),
                mkPrediction("c", "b"),
                mkPrediction("b", "b"),
                mkPrediction("b", "c"),
                mkPrediction("a", "b")
        );
        ImmutableOutputInfo<Label> domain = mkDomain(predictions);
        LabelConfusionMatrix cm = new LabelConfusionMatrix(domain, predictions);

        Label a = label("a");
        Label b = label("b");
        Label c = label("c");

        assertEquals(1, cm.confusion(a, a));
        assertEquals(1, cm.confusion(b, c));
        assertEquals(1, cm.confusion(c, b));

        assertEquals(1, cm.tp(a));
        assertEquals(0, cm.fp(a));
        assertEquals(3, cm.tn(a));
        assertEquals(1, cm.fn(a));
        assertEquals(2, cm.support(a));

        assertEquals(1, cm.tp(b));
        assertEquals(2, cm.fp(b));
        assertEquals(1, cm.tn(b));
        assertEquals(1, cm.fn(b));
        assertEquals(2, cm.support(b));

        assertEquals(0, cm.tp(c));
        assertEquals(1, cm.fp(c));
        assertEquals(3, cm.tn(c));
        assertEquals(1, cm.fn(c));
        assertEquals(1, cm.support(c));

        assertEquals(5, cm.support());
        String cmToString = cm.toString();
        assertEquals("       a   b   c\n" + 
        			 "a      1   1   0\n" +
        			 "b      0   1   1\n" + 
        			 "c      0   1   0\n", cmToString);

    }

}