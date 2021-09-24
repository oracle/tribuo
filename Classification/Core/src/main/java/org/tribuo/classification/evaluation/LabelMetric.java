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

import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.evaluation.metrics.EvaluationMetric;
import org.tribuo.evaluation.metrics.MetricContext;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.sequence.SequenceModel;

import java.util.List;
import java.util.Objects;
import java.util.function.ToDoubleBiFunction;

/**
 * A {@link EvaluationMetric} for {@link Label}s which calculates the value based on a
 * {@link ConfusionMatrix}.
 */
public class LabelMetric implements EvaluationMetric<Label, LabelMetric.Context> {

    private final MetricTarget<Label> tgt;
    private final String name;
    private final ToDoubleBiFunction<MetricTarget<Label>, Context> impl;

    /**
     * Construct a new {@code LabelMetric} for the supplied metric target,
     * using the supplied function.
     * @param tgt The metric target.
     * @param name The name of the metric.
     * @param impl The implementing function.
     */
    public LabelMetric(MetricTarget<Label> tgt, String name,
                       ToDoubleBiFunction<MetricTarget<Label>, Context> impl) {
        this.tgt = tgt;
        this.name = name;
        this.impl = impl;
    }

    @Override
    public double compute(LabelMetric.Context context) {
        return impl.applyAsDouble(tgt, context);
    }

    @Override
    public MetricTarget<Label> getTarget() {
        return tgt;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LabelMetric that = (LabelMetric) o;
        return Objects.equals(tgt, that.tgt) &&
                Objects.equals(name, that.name) &&
                Objects.equals(impl, that.impl);
    }

    @Override
    public int hashCode() {
        return Objects.hash(tgt, name, impl);
    }

    @Override
    public String toString() {
        return "LabelMetric{" +
                "target=" + tgt +
                ", name='" + name +
                '}';
    }

    @Override
    public Context createContext(Model<Label> model, List<Prediction<Label>> predictions) {
        return new Context(model, predictions);
    }

    /**
     * The context for a {@link LabelMetric} is a {@link ConfusionMatrix}.
     */
    public static final class Context extends MetricContext<Label> {

        private final ConfusionMatrix<Label> cm;

        /**
         * Constructs a context and compute the confusion matrix using the specified model and predictions.
         * @param model The model.
         * @param predictions The predictions.
         */
        public Context(Model<Label> model, List<Prediction<Label>> predictions) {
            super(model, predictions);
            this.cm = new LabelConfusionMatrix(model.getOutputIDInfo(), predictions);
        }

        /**
         * Constructs a context and compute the confusion matrix using the specified model and predictions.
         * @param model The model.
         * @param predictions The predictions.
         */
        public Context(SequenceModel<Label> model, List<Prediction<Label>> predictions) {
            super(model, predictions);
            this.cm = new LabelConfusionMatrix(model.getOutputIDInfo(), predictions);
        }

        /**
         * Gets the confusion matrix.
         * @return The confusion matrix.
         */
        public ConfusionMatrix<Label> getCM() {
            return cm;
        }
    }
}