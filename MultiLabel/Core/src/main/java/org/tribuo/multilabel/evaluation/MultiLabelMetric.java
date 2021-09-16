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

package org.tribuo.multilabel.evaluation;

import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.evaluation.ConfusionMatrix;
import org.tribuo.evaluation.metrics.EvaluationMetric;
import org.tribuo.evaluation.metrics.MetricContext;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.multilabel.MultiLabel;

import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;

/**
 * A {@link EvaluationMetric} for evaluating {@link MultiLabel} problems.
 * The sufficient statistics used must be held in a {@link ConfusionMatrix}.
 */
public class MultiLabelMetric implements EvaluationMetric<MultiLabel, MultiLabelMetric.Context> {

    private final MetricTarget<MultiLabel> target;
    private final String name;
    private final BiFunction<MetricTarget<MultiLabel>, Context, Double> impl;

    /**
     * Constructs a multi-label metric.
     * @param target The metric target.
     * @param name The name of the metric.
     * @param impl The implementing function for this metric.
     */
    public MultiLabelMetric(MetricTarget<MultiLabel> target, String name, BiFunction<MetricTarget<MultiLabel>, Context, Double> impl) {
        this.target = target;
        this.name = name;
        this.impl = impl;
    }

    @Override
    public MetricTarget<MultiLabel> getTarget() {
        return target;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public double compute(Context context) {
        return impl.apply(target, context);
    }

    @Override
    public String toString() {
        return "MultiLabelMetric{" +
                "target=" + target +
                ", name='" + name + '\'' +
                ", impl=" + impl +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MultiLabelMetric that = (MultiLabelMetric) o;
        return Objects.equals(target, that.target) &&
                Objects.equals(name, that.name) &&
                Objects.equals(impl, that.impl);
    }

    @Override
    public int hashCode() {
        return Objects.hash(target, name, impl);
    }

    @Override
    public Context createContext(Model<MultiLabel> model, List<Prediction<MultiLabel>> predictions) {
        return buildContext(model, predictions);
    }

    static final class Context extends MetricContext<MultiLabel> {
        private final ConfusionMatrix<MultiLabel> cm;

        Context(Model<MultiLabel> model, List<Prediction<MultiLabel>> predictions, ConfusionMatrix<MultiLabel> cm) {
            super(model, predictions);
            this.cm = cm;
        }

        ConfusionMatrix<MultiLabel> getCM() {
            return cm;
        }
    }

    static Context buildContext(Model<MultiLabel> model, List<Prediction<MultiLabel>> predictions) {
        ConfusionMatrix<MultiLabel> cm = new MultiLabelConfusionMatrix(model, predictions);
        return new Context(model, predictions, cm);
    }

}