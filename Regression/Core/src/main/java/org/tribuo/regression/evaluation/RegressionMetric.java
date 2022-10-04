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

package org.tribuo.regression.evaluation;

import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.evaluation.metrics.EvaluationMetric;
import org.tribuo.evaluation.metrics.MetricContext;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.regression.Regressor;

import java.util.List;
import java.util.function.ToDoubleBiFunction;

/**
 * A {@link EvaluationMetric} for {@link Regressor}s which calculates the metric based on a
 * the true values and the predicted values.
 */
public class RegressionMetric implements EvaluationMetric<Regressor, RegressionMetric.Context> {

    private final MetricTarget<Regressor> tgt;
    private final String name;
    private final ToDoubleBiFunction<MetricTarget<Regressor>, Context> impl;
    private final boolean useExampleWeights;

    /**
     * Construct a new {@code RegressionMetric} for the supplied metric target,
     * using the supplied function. This does not use example weights.
     * @param tgt The metric target.
     * @param name The name of the metric.
     * @param impl The implementing function.
     */
    public RegressionMetric(MetricTarget<Regressor> tgt,
                            String name,
                            ToDoubleBiFunction<MetricTarget<Regressor>, Context> impl) {
        this(tgt, name, impl, false);
    }

    /**
     * Construct a new {@code RegressionMetric} for the supplied metric target,
     * using the supplied function.
     * @param tgt The metric target.
     * @param name The name of the metric.
     * @param impl The implementing function.
     * @param useExampleWeights If true then the example weights are used to scale the example importance.
     */
    public RegressionMetric(MetricTarget<Regressor> tgt,
                            String name,
                            ToDoubleBiFunction<MetricTarget<Regressor>, Context> impl,
                            boolean useExampleWeights) {
        this.tgt = tgt;
        this.name = name;
        this.impl = impl;
        this.useExampleWeights = useExampleWeights;
    }

    @Override
    public double compute(Context context) {
        return impl.applyAsDouble(tgt, context);
    }

    @Override
    public MetricTarget<Regressor> getTarget() {
        return tgt;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public Context createContext(Model<Regressor> model, List<Prediction<Regressor>> predictions) {
        return new Context(model, predictions, useExampleWeights);
    }

    /**
     * The {@link MetricContext} for a {@link Regressor} is each true value and each predicted value for all dimensions.
     */
    static class Context extends MetricContext<Regressor> {
        private final RegressionSufficientStatistics memo;

        Context(Model<Regressor> model, List<Prediction<Regressor>> predictions, boolean useExampleWeights) {
            super(model, predictions);
            this.memo = new RegressionSufficientStatistics(model.getOutputIDInfo(),predictions, useExampleWeights);
        }

        RegressionSufficientStatistics getMemo() {
            return memo;
        }
    }
}