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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.evaluation.metrics.EvaluationMetric;
import org.tribuo.evaluation.metrics.MetricContext;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.util.Util;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;


/**
 * Aggregates metrics from a list of evaluations, or a list of models and datasets.
 */
public final class EvaluationAggregator {

    // singleton
    private EvaluationAggregator() {}

    /**
     * Summarize performance w.r.t. metric across several models on a single dataset.
     * @param metric The metric to summarise.
     * @param models The models to evaluate.
     * @param dataset The dataset to evaluate.
     * @param <T> The output type.
     * @param <C> The context type used for this metric.
     * @return The descriptive statistics for this metric summary.
     */
    public static <T extends Output<T>,
            C extends MetricContext<T>> DescriptiveStats summarize(EvaluationMetric<T,C> metric, List<? extends Model<T>> models, Dataset<T> dataset) {
        DescriptiveStats summary = new DescriptiveStats();
        for (Model<T> model : models) {
            C ctx = metric.createContext(model, dataset);
            double value = metric.compute(ctx);
            summary.addValue(value);
        }
        return summary;
    }

    /**
     * Summarize performance using the supplied evaluator across several models on one dataset.
     * @param evaluator The evaluator to use.
     * @param models The models to evaluate.
     * @param dataset The dataset to evaluate.
     * @param <T> The output type.
     * @param <R> The evaluation type.
     * @return Descriptive statistics for each metric in the evaluator.
     */
    public static <T extends Output<T>,
            R extends Evaluation<T>> Map<MetricID<T>, DescriptiveStats> summarize(Evaluator<T,R> evaluator, List<? extends Model<T>> models, Dataset<T> dataset) {
        List<R> evals = models.stream().map(model -> evaluator.evaluate(model, dataset)).collect(Collectors.toList());
        return summarize(evals);
    }

    /**
     * Summarize a model's performance w.r.t. a metric across several datasets.
     *
     * @param metric The metric to evaluate.
     * @param model The model to evaluate.
     * @param datasets The datasets to evaluate.
     * @param <T> The output type.
     * @param <C> The metric context type.
     * @return Descriptive statistics for the metric across the datasets.
     */
    public static <T extends Output<T>,
            C extends MetricContext<T>> DescriptiveStats summarize(EvaluationMetric<T,C> metric, Model<T> model, List<? extends Dataset<T>> datasets) {
        DescriptiveStats summary = new DescriptiveStats();
        for (Dataset<T> dataset : datasets) {
            C ctx = metric.createContext(model, dataset);
            double value = metric.compute(ctx);
            summary.addValue(value);
        }
        return summary;
    }

    /**
     * Summarize model performance on dataset across several metrics.
     * @param metrics The metrics to evaluate.
     * @param model The model to evaluate them on.
     * @param dataset The dataset to evaluate them on.
     * @param <T> The output type.
     * @param <C> The metric context type.
     * @return The descriptive statistics for the metrics.
     */
    public static <T extends Output<T>,
            C extends MetricContext<T>> DescriptiveStats summarize(List<? extends EvaluationMetric<T,C>> metrics, Model<T> model, Dataset<T> dataset) {
        List<Prediction<T>> predictions = model.predict(dataset);
        DescriptiveStats summary = new DescriptiveStats();
        for (EvaluationMetric<T,C> metric : metrics) {
            C ctx = metric.createContext(model, predictions);
            double value = metric.compute(ctx);
            summary.addValue(value);
        }
        return summary;
    }

    /**
     * Summarize model performance on dataset across several metrics.
     * @param metrics The metrics to evaluate.
     * @param model The model to evaluate them on.
     * @param predictions The predictions to evaluate.
     * @param <T> The output type.
     * @param <C> The metric context type.
     * @return The descriptive statistics for the metrics.
     */
    public static <T extends Output<T>,
            C extends MetricContext<T>> DescriptiveStats summarize(List<? extends EvaluationMetric<T,C>> metrics, Model<T> model, List<Prediction<T>> predictions) {
        DescriptiveStats summary = new DescriptiveStats();
        for (EvaluationMetric<T,C> metric : metrics) {
            C ctx = metric.createContext(model, predictions);
            double value = metric.compute(ctx);
            summary.addValue(value);
        }
        return summary;
    }

    /**
     * Summarize performance according to evaluator for a single model across several datasets.
     * @param evaluator The evaluator to use.
     * @param model The model to evaluate.
     * @param datasets The datasets to evaluate across.
     * @param <T> The output type.
     * @param <R> The evaluation type.
     * @return The descriptive statistics for each metric.
     */
    public static <T extends Output<T>,
            R extends Evaluation<T>> Map<MetricID<T>, DescriptiveStats> summarize(Evaluator<T,R> evaluator, Model<T> model, List<? extends Dataset<T>> datasets) {
        List<R> evals = datasets.stream().map(data -> evaluator.evaluate(model, data)).collect(Collectors.toList());
        return summarize(evals);
    }

    /**
     * Summarize all fields of a list of evaluations.
     * @param evaluations The evaluations to summarize.
     * @param <T> The output type.
     * @param <R> The evaluation type.
     * @return The descriptive statistics for each metric.
     */
    public static <T extends Output<T>, R extends Evaluation<T>> Map<MetricID<T>, DescriptiveStats> summarize(List<R> evaluations) {
        Map<MetricID<T>, DescriptiveStats> results = new HashMap<>();
        for (R evaluation : evaluations) {
            for (Map.Entry<MetricID<T>, Double> kv : evaluation.asMap().entrySet()) {
                MetricID<T> key = kv.getKey();
                DescriptiveStats summary = results.getOrDefault(key, new DescriptiveStats());
                summary.addValue(kv.getValue());
                results.put(key, summary);
            }
        }
        return results;
    }

    /**
     * Summarize all fields of a list of evaluations produced by {@link CrossValidation}.
     * @param evaluations The evaluations to summarize.
     * @param <T> The output type.
     * @param <R> The evaluation type.
     * @return The descriptive statistics for each metric.
     */
    public static <T extends Output<T>, R extends Evaluation<T>> Map<MetricID<T>, DescriptiveStats> summarizeCrossValidation(List<Pair<R, Model<T>>> evaluations) {
        Map<MetricID<T>, DescriptiveStats> results = new HashMap<>();
        for (Pair<R,Model<T>> pair : evaluations) {
            R evaluation = pair.getA();
            for (Map.Entry<MetricID<T>, Double> kv : evaluation.asMap().entrySet()) {
                MetricID<T> key = kv.getKey();
                DescriptiveStats summary = results.getOrDefault(key, new DescriptiveStats());
                summary.addValue(kv.getValue());
                results.put(key, summary);
            }
        }
        return results;
    }

    /**
     * Summarize a single field of an evaluation across several evaluations.
     *
     * @param evaluations the evaluations
     * @param fieldGetter the getter for the field to summarize
     * @param <T> the type of the output
     * @param <R> the type of the evaluation
     * @return a descriptive stats summary of field
     */
    public static <T extends Output<T>, R extends Evaluation<T>> DescriptiveStats summarize(List<R> evaluations, ToDoubleFunction<R> fieldGetter) {
        DescriptiveStats summary = new DescriptiveStats();
        for (R evaluation : evaluations) {
            double value = fieldGetter.applyAsDouble(evaluation);
            summary.addValue(value);
        }
        return summary;
    }

    /**
     * Calculates the argmax of a metric across the supplied models (i.e., the index of the model which performed the best).
     * @param metric The metric to evaluate.
     * @param models The models to evaluate across.
     * @param dataset The dataset to evaluate on.
     * @param <T> The output type.
     * @param <C> The metric context.
     * @return The maximum value and it's index in the models list.
     */
    public static <T extends Output<T>,
            C extends MetricContext<T>> Pair<Integer, Double> argmax(EvaluationMetric<T,C> metric, List<? extends Model<T>> models, Dataset<T> dataset) {
        List<Double> values = models.stream()
                .map(model -> metric.compute(metric.createContext(model, dataset)))
                .collect(Collectors.toList());
        return Util.argmax(values);
    }

    /**
     * Calculates the argmax of a metric across the supplied datasets.
     * @param metric The metric to evaluate.
     * @param model The model to evaluate on.
     * @param datasets The datasets to evaluate across.
     * @param <T> The output type.
     * @param <C> The metric context.
     * @return The maximum value and it's index in the datasets list.
     */
    public static <T extends Output<T>,
            C extends MetricContext<T>> Pair<Integer, Double> argmax(EvaluationMetric<T,C> metric, Model<T> model, List<? extends Dataset<T>> datasets) {
        List<Double> values = datasets.stream()
                .map(dataset -> metric.compute(metric.createContext(model, dataset)))
                .collect(Collectors.toList());
        return Util.argmax(values);
    }

    /**
     * Calculates the argmax of a metric across the supplied evaluations.
     * @param evaluations The evaluations.
     * @param getter The function to extract a value from the evaluation.
     * @param <T> The output type.
     * @param <R> The evaluation type.
     * @return The maximum value and it's index in the evaluations list.
     */
    public static <T extends Output<T>, R extends Evaluation<T>> Pair<Integer, Double> argmax(List<R> evaluations, Function<R, Double> getter) {
        List<Double> values = evaluations.stream().map(getter).collect(Collectors.toList());
        return Util.argmax(values);
    }

}