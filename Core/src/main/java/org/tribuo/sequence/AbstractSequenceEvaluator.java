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

package org.tribuo.sequence;

import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.evaluation.metrics.EvaluationMetric;
import org.tribuo.evaluation.metrics.MetricContext;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.EvaluationProvenance;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Base class for sequence evaluators.
 */
public abstract class AbstractSequenceEvaluator<
        T extends Output<T>,
        C extends MetricContext<T>,
        E extends SequenceEvaluation<T>,
        M extends EvaluationMetric<T, C>> implements SequenceEvaluator<T, E> {

    /**
     * Produces an evaluation for the supplied model and dataset, by calling {@link SequenceModel#predict}
     * to create the predictions, then aggregating the appropriate statistics.
     * @param model The model to use.
     * @param dataset The dataset to make predictions for.
     * @return An evaluation of the dataset on the model.
     */
    @Override
    public final E evaluate(SequenceModel<T> model, SequenceDataset<T> dataset) {
        //
        // Run the model against the dataset to get predictions
        List<List<Prediction<T>>> predictions = model.predict(dataset);
        return evaluate(model, predictions, dataset.getProvenance());
    }

    /**
     * Produces an evaluation for the supplied model and datasource, by calling {@link SequenceModel#predict}
     * to create the predictions, then aggregating the appropriate statistics.
     * @param model The model to use.
     * @param datasource The datasource to make predictions for.
     * @return An evaluation of the datasource on the model.
     */
    @Override
    public final E evaluate(SequenceModel<T> model, SequenceDataSource<T> datasource) {
        //
        // Run the model against the datasource to get predictions
        List<List<Prediction<T>>> predictions = model.predict(datasource);
        return evaluate(model, predictions, datasource.getProvenance());
    }

    // "template method"

    /**
     * Produces an evaluation for the supplied model and predictions by aggregating the appropriate statistics.
     * <p>
     * Warning, this method cannot validate that the predictions were returned by the model in question.
     * @param model The model to use.
     * @param predictions The predictions to use.
     * @param dataProvenance The provenance of the test data.
     * @return An evaluation of the predictions.
     */
    @Override
    public final E evaluate(SequenceModel<T> model, List<List<Prediction<T>>> predictions, DataProvenance dataProvenance) {
        //
        // Create the provenance for the model and dataset
        EvaluationProvenance provenance = new EvaluationProvenance(model.getProvenance(), dataProvenance);
        //
        // Create an evaluation context. The context stores all the information needed by the list of metrics plus might
        // cache intermediate computation relevant to multiple metrics (e.g., a pre-computed confusion matrix might be stored in 'context')
        C context = createContext(model, predictions);
        //
        // "MODEL": Build the list of metrics to compute.
        Set<? extends EvaluationMetric<T, C>> metrics = createMetrics(model);
        //
        // "CONTROLLER": For each metric in the list, compute the result.
        Map<MetricID<T>, Double> results = computeResults(context, metrics);
        //
        // "VIEW": Create an evaluation to store the results and provide a "view" of the results to users
        return createEvaluation(context, results, provenance);
    }

    /**
     * Computes each metric given the context.
     * @param ctx The metric context (i.e., the sufficient statistics).
     * @param metrics The metrics to compute.
     * @return The value of each requested metric.
     */
    protected Map<MetricID<T>, Double> computeResults(C ctx, Set<? extends EvaluationMetric<T, C>> metrics) {
        Map<MetricID<T>, Double> results = new HashMap<>();
        for (EvaluationMetric<T, C> metric : metrics) {
            MetricID<T> id = metric.getID();
            double value = metric.compute(ctx);
            results.put(id, value);
        }
        return results;
    }

    /**
     * Creates the appropriate set of metrics for this model, by querying for it's {@link org.tribuo.OutputInfo}.
     * @param model The model to inspect.
     * @return The set of metrics.
     */
    protected abstract Set<M> createMetrics(SequenceModel<T> model);

    //
    // Note: the following two methods are abstract (plus the 'C' type parameter) to make memoization work smoothly, basically.

    /**
     * Create the context needed for evaluation. The context might store global properties or cache computation.
     * @param model the model that will be evaluated
     * @param predictions the predictions that will be evaluated
     * @return the context for this model and its predictions
     */
    protected abstract C createContext(SequenceModel<T> model, List<List<Prediction<T>>> predictions);

    /**
     * Create an evaluation for the given results
     * @param context the context that was used to compute these results
     * @param results the results
     * @param provenance the provenance of the results (including information about the model and dataset)
     * @return the evaluation
     */
    protected abstract E createEvaluation(C context,
                                          Map<MetricID<T>, Double> results,
                                          EvaluationProvenance provenance);
}