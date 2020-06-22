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
import org.tribuo.provenance.DataProvenance;

import java.util.List;

/**
 * An evaluation factory which produces immutable {@link SequenceEvaluation}s of a given {@link SequenceDataset} using the given {@link SequenceModel}.
 * @param <T> The output type.
 * @param <E> The evaluation type.
 */
public interface SequenceEvaluator<T extends Output<T>, E extends SequenceEvaluation<T>> {

    /**
     * Evaluates the dataset using the supplied model, returning an immutable evaluation.
     * @param model The model to use.
     * @param dataset The dataset to evaluate.
     * @return An evaluation.
     */
    public E evaluate(SequenceModel<T> model, SequenceDataset<T> dataset);

    /**
     * Evaluates the datasource using the supplied model, returning an immutable evaluation.
     * @param model The model to use.
     * @param datasource The datasource to evaluate.
     * @return An evaluation.
     */
    public E evaluate(SequenceModel<T> model, SequenceDataSource<T> datasource);

    /**
     * Evaluates the supplied model and predictions by aggregating the appropriate statistics.
     * <p>
     * Warning, this method cannot validate that the predictions were returned by the model in question.
     * @param model The model to use.
     * @param predictions The predictions to use.
     * @param dataProvenance The provenance of the test data.
     * @return An evaluation of the predictions.
     */
    public E evaluate(SequenceModel<T> model, List<List<Prediction<T>>> predictions, DataProvenance dataProvenance);

}
