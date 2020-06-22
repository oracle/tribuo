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

package org.tribuo.evaluation;

import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.provenance.EvaluationProvenance;

import java.util.List;
import java.util.Map;

/**
 * An immutable evaluation of a specific model and dataset.
 * @param <T> The output type.
 */
public interface Evaluation<T extends Output<T>> extends Provenancable<EvaluationProvenance> {

    /**
     * Get a map of all the metrics stored in this evaluation. The keys are metric id's and the values are their
     * corresponding computed results.
     *
     * @return a map of all stored results
     */
    public Map<MetricID<T>, Double> asMap();

    /**
     * Gets the value associated with the specific metric. Throws {@link IllegalArgumentException} if
     * the metric is unknown.
     * @param key The metric to lookup.
     * @return The value for that metric.
     */
    public default double get(MetricID<T> key) {
        Double value = asMap().get(key);
        if (value == null) {
            throw new IllegalArgumentException("Metric value not found: " + key.toString());
        }
        return value;
    }

    /**
     * Gets the predictions stored in this evaluation.
     * @return The predictions.
     */
    public List<Prediction<T>> getPredictions();

}