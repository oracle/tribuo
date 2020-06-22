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

import org.tribuo.Output;

import java.util.Objects;
import java.util.Optional;

/**
 * Used by a given {@link EvaluationMetric} to determine whether it should compute its value for a specific {@link Output} value
 * or whether it should average them.
 *
 * @param <T> The {@link Output} type.
 */
public class MetricTarget<T extends Output<T>> {

    private final T target;
    private final EvaluationMetric.Average avg;
    // TODO none value? what about cases like balanced error rate / ami?
    // - rename Average to Aggregate? then return Aggregate "all"/"domainwise"
    //   sometimes?

    /**
     * Builds a metric target for an output.
     * @param target The output to target.
     */
    public MetricTarget(T target) {
        this.target = target;
        this.avg = null;
    }

    /**
     * Builds a metric target for an average.
     * @param avg The average to compute.
     */
    public MetricTarget(EvaluationMetric.Average avg) {
        this.target = null;
        this.avg = avg;
    }

    /**
     * Returns the Output this metric targets, or {@link Optional#empty} if it's an average.
     * @return The output this metric targets, or {@link Optional#empty}.
     */
    public Optional<T> getOutputTarget() { return Optional.ofNullable(target); }

    /**
     * Returns the average this metric computes, or {@link Optional#empty} if it targets an output.
     * @return The average this metric computes, or {@link Optional#empty}.
     */
    public Optional<EvaluationMetric.Average> getAverageTarget() { return Optional.ofNullable(avg); }

    @Override
    public String toString() {
        if (getOutputTarget().isPresent()) {
            return String.format("MetricTarget{output=%s}", getOutputTarget().get());
        } else {
            return String.format("MetricTarget{average=%s}", getAverageTarget().get().name());
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MetricTarget<?> that = (MetricTarget<?>) o;
        return Objects.equals(target, that.target) &&
                avg == that.avg;
    }

    @Override
    public int hashCode() {
        return Objects.hash(target, avg);
    }

    private static final MetricTarget<?> macroTarget = new MetricTarget<>(EvaluationMetric.Average.MACRO);
    private static final MetricTarget<?> microTarget = new MetricTarget<>(EvaluationMetric.Average.MICRO);

    /**
     * Get the singleton {@code MetricTarget} which contains the {@link EvaluationMetric.Average#MACRO}.
     *
     * @param <U> The output type of the {@code MetricTarget}
     * @return The {@code MetricTarget} that provides a macro average.
     */
    @SuppressWarnings("unchecked")
    public static <U extends Output<U>> MetricTarget<U> macroAverageTarget() {
        return (MetricTarget<U>) macroTarget;
    }

    /**
     * Get the singleton {@code MetricTarget} which contains the {@link EvaluationMetric.Average#MICRO}.
     *
     * @param <U> The output type of the {@code MetricTarget}
     * @return The {@code MetricTarget} that provides a micro average.
     */
    @SuppressWarnings("unchecked")
    public static <U extends Output<U>> MetricTarget<U> microAverageTarget() {
        return (MetricTarget<U>) microTarget;
    }
}