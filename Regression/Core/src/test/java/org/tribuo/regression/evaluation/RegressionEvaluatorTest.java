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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.baseline.DummyRegressionTrainer;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RegressionEvaluatorTest {

    @Test
    public void testSingleDim() {
        // TODO compute by hand and make sure evaluation gets correct results
        Pair<Dataset<Regressor>,Dataset<Regressor>> pair = RegressionDataGenerator.denseTrainTest();
        //
        // trainer is a noop: just produces a model that predicts 0 for everything
        Trainer<Regressor> trainer = DummyRegressionTrainer.createConstantTrainer(0d);
        Model<Regressor> model = trainer.train(pair.getA());
        List<Prediction<Regressor>> predictions = model.predict(pair.getB());

        RegressionEvaluation evaluation = new RegressionEvaluator()
                .evaluate(model, predictions, pair.getB().getProvenance());

        Regressor variable = model.getOutputIDInfo().getOutput(0);
        assertEquals(27.504590525946757, evaluation.rmse(variable));
        assertEquals(21.275, evaluation.mae(variable));
        assertEquals(-1.4895033555942687, evaluation.r2(variable));
        assertEquals(0.0, evaluation.explainedVariance(variable));
    }

    @Test
    public void testSingleDimWeighted() {
        // TODO compute by hand and make sure evaluation gets correct results
        Pair<Dataset<Regressor>,Dataset<Regressor>> pair = RegressionDataGenerator.denseTrainTest();
        //
        // trainer is a noop: just produces a model that predicts 0 for everything
        Trainer<Regressor> trainer = DummyRegressionTrainer.createConstantTrainer(0d);
        Model<Regressor> model = trainer.train(pair.getA());
        //
        // set random example weights
        Random rng = new Random(0L);
        for (Example<Regressor> example : pair.getB()) {
            example.setWeight(rng.nextFloat());
        }
        List<Prediction<Regressor>> predictions = model.predict(pair.getB());

        RegressionEvaluation evaluation = new RegressionEvaluator(true)
                .evaluate(model, predictions, pair.getB().getProvenance());

        //out.println(evaluation);

        Regressor variable = model.getOutputIDInfo().getOutput(0);
        assertEquals(26.67414414121881, evaluation.rmse(variable));
        assertEquals(19.57851730419376, evaluation.mae(variable));
        assertEquals(-1.1679711170396367, evaluation.r2(variable));
        assertEquals(0.0, evaluation.explainedVariance(variable));
    }

}