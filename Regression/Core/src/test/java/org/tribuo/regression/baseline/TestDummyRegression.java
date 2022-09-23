/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression.baseline;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.baseline.DummyRegressionTrainer;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.fail;

/**
 *
 */
public class TestDummyRegression {

    private static final Trainer<Regressor> constant = DummyRegressionTrainer.createConstantTrainer(9.0);
    private static final Trainer<Regressor> mean = DummyRegressionTrainer.createMeanTrainer();
    private static final Trainer<Regressor> median = DummyRegressionTrainer.createMedianTrainer();
    private static final Trainer<Regressor> quartile = DummyRegressionTrainer.createQuartileTrainer(0.9);
    private static final Trainer<Regressor> gaussian = DummyRegressionTrainer.createGaussianTrainer(1L);

    private static final List<Trainer<Regressor>> trainers = Arrays.asList(constant,mean,median,quartile,gaussian);

    public void testDummyRegression(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testModelSave) {
        for (Trainer<Regressor> t : trainers) {
            Model<Regressor> m = t.train(p.getA());
            RegressionEvaluator evaluator = new RegressionEvaluator();
            RegressionEvaluation evaluation = evaluator.evaluate(m,p.getB());
            if (testModelSave) {
                Helpers.testModelSerialization(m,Regressor.class);
                Helpers.testModelProtoSerialization(m, Regressor.class, p.getB());
            }
        }
    }

    @Test
    public void testQuartileException() {
        try {
            Trainer<Regressor> t = DummyRegressionTrainer.createQuartileTrainer(-1.0);
            fail("Should have thrown IllegalArgumentException");
        } catch (IllegalArgumentException e) { }
        try {
            Trainer<Regressor> t = DummyRegressionTrainer.createQuartileTrainer(1.1);
            fail("Should have thrown IllegalArgumentException");
        } catch (IllegalArgumentException e) { }
        try {
            Trainer<Regressor> t = DummyRegressionTrainer.createQuartileTrainer(Double.POSITIVE_INFINITY);
            fail("Should have thrown IllegalArgumentException");
        } catch (IllegalArgumentException e) { }
        try {
            Trainer<Regressor> t = DummyRegressionTrainer.createQuartileTrainer(Double.NaN);
            fail("Should have thrown IllegalArgumentException");
        } catch (IllegalArgumentException e) { }
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        testDummyRegression(p,true);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testDummyRegression(p,false);
    }
}
