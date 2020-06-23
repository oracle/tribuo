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

package org.tribuo.classification.sequence;

import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelMetric;
import org.tribuo.classification.evaluation.LabelMetrics;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.provenance.EvaluationProvenance;
import org.tribuo.sequence.AbstractSequenceEvaluator;
import org.tribuo.sequence.SequenceModel;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A sequence evaluator for labels.
 */
public class LabelSequenceEvaluator extends AbstractSequenceEvaluator<Label, LabelMetric.Context, LabelSequenceEvaluation, LabelMetric> {

    @Override
    protected Set<LabelMetric> createMetrics(SequenceModel<Label> model) {
        Set<LabelMetric> metrics = new HashSet<>();
        //
        // Populate labelwise values
        for (Label label : model.getOutputIDInfo().getDomain()) {
            MetricTarget<Label> tgt = new MetricTarget<>(label);
            metrics.add(LabelMetrics.TP.forTarget(tgt));
            metrics.add(LabelMetrics.FP.forTarget(tgt));
            metrics.add(LabelMetrics.TN.forTarget(tgt));
            metrics.add(LabelMetrics.FN.forTarget(tgt));
            metrics.add(LabelMetrics.PRECISION.forTarget(tgt));
            metrics.add(LabelMetrics.RECALL.forTarget(tgt));
            metrics.add(LabelMetrics.F1.forTarget(tgt));
            metrics.add(LabelMetrics.ACCURACY.forTarget(tgt));
        }

        //
        // Populate averaged values.
        MetricTarget<Label> micro = MetricTarget.microAverageTarget();
        metrics.add(LabelMetrics.TP.forTarget(micro));
        metrics.add(LabelMetrics.FP.forTarget(micro));
        metrics.add(LabelMetrics.TN.forTarget(micro));
        metrics.add(LabelMetrics.FN.forTarget(micro));
        metrics.add(LabelMetrics.PRECISION.forTarget(micro));
        metrics.add(LabelMetrics.RECALL.forTarget(micro));
        metrics.add(LabelMetrics.F1.forTarget(micro));
        metrics.add(LabelMetrics.ACCURACY.forTarget(micro));

        MetricTarget<Label> macro = MetricTarget.macroAverageTarget();
        metrics.add(LabelMetrics.TP.forTarget(macro));
        metrics.add(LabelMetrics.FP.forTarget(macro));
        metrics.add(LabelMetrics.TN.forTarget(macro));
        metrics.add(LabelMetrics.FN.forTarget(macro));
        metrics.add(LabelMetrics.PRECISION.forTarget(macro));
        metrics.add(LabelMetrics.RECALL.forTarget(macro));
        metrics.add(LabelMetrics.F1.forTarget(macro));
        metrics.add(LabelMetrics.ACCURACY.forTarget(macro));

        // Target doesn't matter for balanced error rate, so we just use
        // average.macro as it's the macro average of recalls.
        metrics.add(LabelMetrics.BALANCED_ERROR_RATE.forTarget(macro));

        return metrics;
    }

    @Override
    protected LabelMetric.Context createContext(SequenceModel<Label> model, List<List<Prediction<Label>>> predictions) {
        // Warning this passes a null in as the model.
        return new LabelMetric.Context(model, flattenList(predictions));
    }

    @Override
    protected LabelSequenceEvaluation createEvaluation(LabelMetric.Context ctx,
                                               Map<MetricID<Label>, Double> results,
                                               EvaluationProvenance provenance) {
        return new LabelSequenceEvaluation(results, ctx, provenance);
    }

    private static List<Prediction<Label>> flattenList(List<List<Prediction<Label>>> predictions) {
        List<Prediction<Label>> flatList = new ArrayList<>();

        for (List<Prediction<Label>> list : predictions) {
            flatList.addAll(list);
        }

        return flatList;
    }
}
