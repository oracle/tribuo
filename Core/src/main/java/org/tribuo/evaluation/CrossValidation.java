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
import org.tribuo.Trainer;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A class that does k-fold cross-validation.
 * <p>
 * This splits the data into k pieces, tests on one of them and trains on the rest.
 * <p>
 * It produces a list of {@link Evaluation}s for each of the test sets.
 */
public class CrossValidation<T extends Output<T>, E extends Evaluation<T>> {

    private static final Logger logger = Logger.getLogger(CrossValidation.class.getName());

    private final Trainer<T> trainer;
    private final int numFolds;
    private final Dataset<T> data;
    private final Evaluator<T, E> evaluator;
    private final KFoldSplitter<T> splitter;

    /**
     * Builds a k-fold cross-validation loop.
     * @param trainer the trainer to use.
     * @param data the dataset to split.
     * @param evaluator the evaluator to use.
     * @param k the number of folds.
     */
    public CrossValidation(Trainer<T> trainer,
                           Dataset<T> data,
                           Evaluator<T, E> evaluator,
                           int k) {
        this(trainer, data, evaluator, k, Trainer.DEFAULT_SEED); }

    /**
     * Builds a k-fold cross-validation loop.
     * @param trainer the trainer to use.
     * @param data the dataset to split.
     * @param evaluator the evaluator to use.
     * @param k the number of folds.
     * @param seed The RNG seed.
     */
    public CrossValidation(Trainer<T> trainer,
                           Dataset<T> data,
                           Evaluator<T, E> evaluator,
                           int k,
                           long seed) {
        this.trainer = trainer;
        this.data = data;
        this.evaluator = evaluator;
        this.numFolds = k;
        this.splitter = new KFoldSplitter<>(k, seed);
    }

    /**
     * Returns the number of folds.
     * @return The number of folds.
     */
    public int getK() { return numFolds; }

    /**
     * Performs k fold cross validation, returning the k evaluations.
     * @return The k evaluators one per fold.
     */
    public List<Pair<E, Model<T>>> evaluate() {
        List<Pair<E, Model<T>>> evals = new ArrayList<>();
        Iterator<KFoldSplitter.TrainTestFold<T>> iter = splitter.split(data, true);
        int ct = 0;
        while (iter.hasNext()) {
            logger.log(Level.INFO, "Training for fold " + ct);
            KFoldSplitter.TrainTestFold<T> fold = iter.next();
            Model<T> model = trainer.train(fold.train);
            evals.add(new Pair<>(evaluator.evaluate(model, fold.test), model));
            ct++;
        }
        return evals;
    }
}
