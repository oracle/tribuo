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

import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.DataProvenance;

import java.util.ArrayList;
import java.util.List;

/**
 * An evaluation factory which produces immutable {@link Evaluation}s of a given {@link Dataset} using the given {@link Model}.
 * @param <T> The output type.
 * @param <E> The evaluation type.
 */
public interface Evaluator<T extends Output<T>, E extends Evaluation<T>> {

    /**
     * Evaluates the dataset using the supplied model, returning an immutable {@link Evaluation} of the appropriate type.
     * @param model The model to use.
     * @param dataset The dataset to evaluate.
     * @return An evaluation.
     */
    public E evaluate(Model<T> model, Dataset<T> dataset);

    /**
     * Evaluates the dataset using the supplied model, returning an immutable {@link Evaluation} of the appropriate type.
     * @param model The model to use.
     * @param datasource The data to evaluate.
     * @return An evaluation.
     */
    public E evaluate(Model<T> model, DataSource<T> datasource);

    /**
     * Evaluates the model performance using the supplied predictions, returning an immutable {@link Evaluation}
     * of the appropriate type.
     * <p>
     * It does not validate that the {@code model} produced the supplied {@code predictions}, or that
     * the {@code dataProvenance} matches the input examples. Supplying arguments which do not meet
     * these invariants will produce an invalid Evaluation.
     * </p>
     * @param model The model to use.
     * @param predictions The predictions to evaluate.
     * @param dataProvenance The provenance of the predicted dataset.
     * @return An evaluation.
     */
    public E evaluate(Model<T> model, List<Prediction<T>> predictions, DataProvenance dataProvenance);

    /**
     * Evaluates the model performance using the supplied predictions, returning an immutable {@link Evaluation}
     * of the appropriate type.
     * <p>
     * This method is used when the predictions do not contain the correct ground truth labels (e.g., if they
     * were collected separately from the examples constructed for prediction). First it creates a new set of
     * predictions, containing the same examples with the matched ground truth prediction.
     * Then it calls {@link Evaluator#evaluate(Model, List, DataProvenance)} with the updated predictions.
     * <p>
     * It does not validate that the {@code model} produced the supplied {@code predictions}, or that
     * the {@code dataProvenance} matches the input examples. Supplying arguments which do not meet
     * these invariants will produce an invalid Evaluation.
     * @param model The model to use.
     * @param predictions The predictions to evaluate.
     * @param groundTruth The ground truth outputs to use.
     * @param dataProvenance The provenance of the predicted dataset.
     * @return An evaluation.
     */
    default public E evaluate(Model<T> model, List<Prediction<T>> predictions, List<T> groundTruth, DataProvenance dataProvenance) {
        if (predictions.size() != groundTruth.size()) {
            throw new IllegalArgumentException(
                    "Predictions and ground truth must be the same length, received predictions.size()="
                            +predictions.size()+", groundTruth.size()="+groundTruth.size());
        }
        List<Prediction<T>> newPredictions = new ArrayList<>(predictions.size());

        for (int i = 0; i < predictions.size(); i++) {
            Prediction<T> curPrediction = predictions.get(i);
            Example<T> curExample = curPrediction.getExample();
            ArrayExample<T> newExample = new ArrayExample<>(groundTruth.get(i), curExample, curExample.getWeight());
            Prediction<T> newPrediction = new Prediction<>(curPrediction,curPrediction.getNumActiveFeatures(),newExample);
            newPredictions.add(newPrediction);
        }

        return evaluate(model,newPredictions,dataProvenance);
    }

    /**
     * Creates an online evaluator that maintains a list of all the predictions it has seen and can evaluate
     * them upon request.
     * @param model The model to use for online evaluation.
     * @param provenance The provenance of the data.
     * @return An online evaluator.
     */
    default public OnlineEvaluator<T,E> createOnlineEvaluator(Model<T> model, DataProvenance provenance) {
        return new OnlineEvaluator<>(this,model,provenance);
    }
}