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

package org.tribuo.evaluation.metrics;

import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;

import java.util.List;

/**
 * A metric that can be calculated for the specified output type.
 * @param <T> The output type.
 * @param <C> The context (information necessary to calculate this metric).
 */
public interface EvaluationMetric<T extends Output<T>, C extends MetricContext<T>> {

    /**
     * Compute the result of this metric from the input context.
     * @param context The context to use.
     * @return The value of the metric.
     */
    public double compute(C context);

    /**
     * The target for this metric instance.
     * @return The metric target.
     */
    public MetricTarget<T> getTarget();

    /**
     * The name of this metric.
     * @return The name.
     */
    public String getName();

    /**
     * The metric ID, a combination of the metric target and metric name.
     * @return The metric ID.
     */
    public default MetricID<T> getID() {
        return new MetricID<>(getTarget(), getName());
    }

    /**
     * Creates the context this metric uses to compute it's value.
     * @param model The model to use.
     * @param predictions The predictions to use.
     * @return The metric context.
     */
    public C createContext(Model<T> model, List<Prediction<T>> predictions);

    /**
     * Creates the metric context used to compute this metric's value,
     * generating {@link org.tribuo.Prediction}s for each {@link org.tribuo.Example} in
     * the supplied dataset.
     * @param model The model to use.
     * @param dataset The dataset to predict outputs for.
     * @return The metric context.
     */
    public default C createContext(Model<T> model, Dataset<T> dataset) {
        return createContext(model, model.predict(dataset));
    }

    /**
     * Specifies what form of average to use for a {@link EvaluationMetric}.
     */
    // Note, if we extend this enum, update MetricTarget with new singletons.
    enum Average {
        /**
         * The macro average.
         */
        MACRO,
        /**
         * The micro average.
         */
        MICRO
    }
}