/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.tribuo.classification.Utils.label;
import static org.tribuo.classification.Utils.mkDomain;
import static org.tribuo.classification.Utils.mkPrediction;
import static org.junit.jupiter.api.Assertions.assertEquals;


public class LabelConfusionMatrixTest {

    @Test
    public void testMulticlass() {
        List<Prediction<Label>> predictions = Arrays.asList(
                mkPrediction("a", "a"),
                mkPrediction("c", "b"),
                mkPrediction("b", "b"),
                mkPrediction("b", "c")
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
        assertEquals(0, cm.fn(a));
        assertEquals(1, cm.support(a));

        assertEquals(1, cm.tp(b));
        assertEquals(1, cm.fp(b));
        assertEquals(1, cm.tn(b));
        assertEquals(1, cm.fn(b));
        assertEquals(2, cm.support(b));

        assertEquals(0, cm.tp(c));
        assertEquals(1, cm.fp(c));
        assertEquals(2, cm.tn(c));
        assertEquals(1, cm.fn(c));
        assertEquals(1, cm.support(c));

        assertEquals(4, cm.support());

        List<Label> lblOrder = new ArrayList<>();
        lblOrder.add(a);
        lblOrder.add(b);
        lblOrder.add(c);

        cm.setLabelOrder(lblOrder);

        String cmToString = cm.toString();
        assertEquals("       a   b   c\n" + 
        			 "a      1   0   0\n" + 
        			 "b      0   1   1\n" + 
        			 "c      0   1   0\n", cmToString);

        lblOrder.clear();
        lblOrder.add(c);
        lblOrder.add(a);
        cm.setLabelOrder(lblOrder);

        cmToString = cm.toString();
        assertEquals("       c   a\n" +
            "c      0   0\n" +
            "a      0   1\n", cmToString);

    }

}