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

import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.evaluation.metrics.EvaluationMetric.Average;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.tribuo.classification.Utils.label;
import static org.tribuo.classification.Utils.mkDomain;
import static org.tribuo.classification.Utils.mkPrediction;
import static org.junit.jupiter.api.Assertions.assertEquals;


public class ConfusionMetricsTest {

    @Test
    public void testAccuracy() {
        List<Prediction<Label>> predictions = Arrays.asList(
                mkPrediction("a", "a"),
                mkPrediction("c", "b"),
                mkPrediction("b", "b"),
                mkPrediction("b", "c")
        );
        ImmutableOutputInfo<Label> domain = mkDomain(predictions);
        LabelConfusionMatrix cm = new LabelConfusionMatrix(domain, predictions);

        assertEquals(1d, ConfusionMetrics.accuracy(label("a"), cm));
        assertEquals(0.5, ConfusionMetrics.accuracy(label("b"), cm));

        assertEquals(0d, cm.tp(label("c")));
        assertEquals(0d, ConfusionMetrics.accuracy(label("c"), cm));

        assertEquals(0.5, ConfusionMetrics.accuracy(Average.MICRO, cm));
        assertEquals(0.5, ConfusionMetrics.accuracy(Average.MACRO, cm));
    }

}