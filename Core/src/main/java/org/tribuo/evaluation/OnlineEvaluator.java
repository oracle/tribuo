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

import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.provenance.DataProvenance;

import java.util.ArrayList;
import java.util.List;

/**
 * An evaluator which aggregates predictions and produces {@link Evaluation}s
 * covering all the {@link Prediction}s it has seen or created.
 * @param <T> The output type.
 * @param <E> The evaluation type.
 */
public final class OnlineEvaluator<T extends Output<T>, E extends Evaluation<T>> {

    private final Evaluator<T,E> evaluator;
    private final Model<T> model;
    private final DataProvenance provenance;

    private final List<Prediction<T>> predictions = new ArrayList<>();

    /**
     * Constructs an {@code OnlineEvaluator} which accumulates predictions.
     * @param evaluator The evaluator to use to make {@link Evaluation}s.
     * @param model The model to use.
     * @param provenance The provenance of the evaluation data.
     */
    public OnlineEvaluator(Evaluator<T,E> evaluator, Model<T> model, DataProvenance provenance) {
        this.evaluator = evaluator;
        this.model = model;
        this.provenance = provenance;
    }

    /**
     * Creates an {@link Evaluation} containing all the current
     * predictions.
     * @return An {@link Evaluation} of the appropriate type.
     */
    public E evaluate() {
        return evaluator.evaluate(model,new ArrayList<>(predictions),provenance);
    }

    /**
     * Feeds the example to the model, records the prediction and returns it.
     * @param example The example to predict.
     * @return The model prediction for this example.
     */
    public synchronized Prediction<T> predictAndObserve(Example<T> example) {
        Prediction<T> cur = model.predict(example);
        predictions.add(cur);
        return cur;
    }

    /**
     * Feeds the examples to the model, records the predictions and returns them.
     * @param examples The examples to predict.
     * @return The model predictions for the supplied examples.
     */
    public synchronized List<Prediction<T>> predictAndObserve(Iterable<Example<T>> examples) {
        List<Prediction<T>> cur = model.predict(examples);
        predictions.addAll(cur);
        return new ArrayList<>(cur);
    }

    /**
     * Records the supplied prediction.
     * @param newPrediction The prediction to record.
     */
    public synchronized void observe(Prediction<T> newPrediction) {
        predictions.add(newPrediction);
    }

    /**
     * Records all the supplied predictions.
     * @param newPredictions The predictions to record.
     */
    public synchronized void observe(List<Prediction<T>> newPredictions) {
        predictions.addAll(newPredictions);
    }
}
