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

package org.tribuo.regression.rtree;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.common.tree.TreeModel;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;
import org.tribuo.test.Helpers;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestCARTJointRegressionTrainer {

    private static final RegressionEvaluator e = new RegressionEvaluator();
    private static final CARTJointRegressionTrainer t = new CARTJointRegressionTrainer();
    private static final CARTJointRegressionTrainer normedt = new CARTJointRegressionTrainer(Integer.MAX_VALUE, true);
    private static final CARTJointRegressionTrainer randomt = new CARTJointRegressionTrainer(Integer.MAX_VALUE, 5, 0.0f,
            0.75f, true, new MeanSquaredError(), false, Trainer.DEFAULT_SEED);

    public static Model<Regressor> testJointRegressionTree(Pair<Dataset<Regressor>, Dataset<Regressor>> p, CARTJointRegressionTrainer trainer) {
        TreeModel<Regressor> m = trainer.train(p.getA());
        RegressionEvaluation evaluation = e.evaluate(m, p.getB());

        Map<String, List<Pair<String, Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());

        return m;
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        Model<Regressor> model = testJointRegressionTree(p, t);
        Helpers.testModelSerialization(model, Regressor.class);
        Helpers.testModelProtoSerialization(model, Regressor.class, p.getB());
    }

    @Test
    public void testNormedDenseData() {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        testJointRegressionTree(p, normedt);
    }

    @Test
    public void testRandomDenseData() {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        testJointRegressionTree(p, randomt);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testJointRegressionTree(p, t);
    }

    @Test
    public void testNormedSparseData() {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testJointRegressionTree(p, normedt);
    }

    @Test
    public void testRandomSparseData() {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testJointRegressionTree(p, randomt);
    }

    public void runInvalidExample(CARTJointRegressionTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
            Model<Regressor> m = trainer.train(p.getA());
            m.predict(RegressionDataGenerator.invalidMultiDimSparseExample());
        });
    }

    @Test
    public void testInvalidExample() {
        runInvalidExample(t);
    }

    @Test
    public void testNormedInvalidExample() {
        runInvalidExample(normedt);
    }

    @Test
    public void testRandomInvalidExample() {
        runInvalidExample(randomt);
    }

    public void runEmptyExample(CARTJointRegressionTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
            Model<Regressor> m = trainer.train(p.getA());
            m.predict(RegressionDataGenerator.emptyMultiDimExample());
        });
    }

    @Test
    public void testEmptyExample() {
        runEmptyExample(t);

    }

    @Test
    public void testNormedEmptyExample() {
        runEmptyExample(normedt);

    }

    @Test
    public void testRandomEmptyExample() {
        runEmptyExample(randomt);
    }


    public void runMultiDenseData(CARTJointRegressionTrainer trainer) {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
        testJointRegressionTree(p, trainer);
    }

    @Test
    public void testMultiDenseData() {
        runMultiDenseData(t);
    }

    @Test
    public void testNormedMultiDenseData() {
        runMultiDenseData(normedt);
    }

    @Test
    public void testRandomMultiDenseData() {
        runMultiDenseData(randomt);
    }

    public void runMultiSparseData(CARTJointRegressionTrainer trainer) {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();
        testJointRegressionTree(p, trainer);
    }

    @Test
    public void testMultiSparseData() {
        runMultiSparseData(t);
    }

    @Test
    public void testNormedMultiSparseData() {
        runMultiSparseData(normedt);
    }

    @Test
    public void testRandomMultiSparseData() {
        runMultiSparseData(randomt);
    }

    public void runMultiInvalidExample(CARTJointRegressionTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
            Model<Regressor> m = trainer.train(p.getA());
            m.predict(RegressionDataGenerator.invalidMultiDimSparseExample());
        });
    }

    @Test
    public void testMultiInvalidExample() {
        runMultiInvalidExample(t);
    }

    @Test
    public void testNormedMultiInvalidExample() {
        runMultiInvalidExample(normedt);
    }

    @Test
    public void testRandomMultiInvalidExample() {
        runMultiInvalidExample(randomt);
    }

    public void runMultiEmptyExample(CARTJointRegressionTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
            Model<Regressor> m = trainer.train(p.getA());
            m.predict(RegressionDataGenerator.emptyMultiDimExample());
        });
    }

    @Test
    public void testMultiEmptyExample() {
        runMultiEmptyExample(t);
    }

    @Test
    public void testNormedMultiEmptyExample() {
        runMultiEmptyExample(normedt);
    }

    @Test
    public void testRandomMultiEmptyExample() {
        runMultiEmptyExample(randomt);
    }

    @Test
    public void testThreeDenseData() {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
        TreeModel<Regressor> llModel = t.train(p.getA());
        RegressionEvaluation llEval = e.evaluate(llModel, p.getB());
        double expectedDim1 = -0.6618655170782572;
        double expectedDim2 = -0.6618655170782572;
        double expectedDim3 = -2.072009095885796;
        double expectedAve = -1.1319133766807703;

        assertEquals(expectedDim1, llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName, Double.NaN)), 1e-6);
        assertEquals(expectedDim2, llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName, Double.NaN)), 1e-6);
        assertEquals(expectedDim3, llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName, Double.NaN)), 1e-6);
        assertEquals(expectedAve, llEval.averageR2(), 1e-6);
        Helpers.testModelProtoSerialization(llModel, Regressor.class, p.getB());

        p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, true);
        llModel = t.train(p.getA());
        llEval = e.evaluate(llModel, p.getB());

        assertEquals(expectedDim1, llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName, Double.NaN)), 1e-6);
        assertEquals(expectedDim2, llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName, Double.NaN)), 1e-6);
        assertEquals(expectedDim3, llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName, Double.NaN)), 1e-6);
        assertEquals(expectedAve, llEval.averageR2(), 1e-6);

    }
}
