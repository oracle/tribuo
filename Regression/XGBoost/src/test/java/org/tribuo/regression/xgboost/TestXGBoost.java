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

package org.tribuo.regression.xgboost;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestXGBoost {

    private static final XGBoostRegressionTrainer t = new XGBoostRegressionTrainer(50);

    private static final RegressionEvaluator e = new RegressionEvaluator();

    public static Model<Regressor> testXGBoost(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = t.train(p.getA());
        e.evaluate(m,p.getB());
        return m;
    }

    @Test
    public void testMultiDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
        Model<Regressor> model = testXGBoost(p);
        Helpers.testModelSerialization(model,Regressor.class);
    }

    @Test
    public void testMultiSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();
        testXGBoost(p);
    }

    @Test
    public void testThreeDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, true);
        Model<Regressor> xgbModel = t.train(p.getA());
        RegressionEvaluation xgbEval = e.evaluate(xgbModel,p.getB());
        double expectedDim1 = -0.6516752237887062;
        double expectedDim2 = -0.6494935599069476;
        double expectedDim3 = -1.4848856827996757;
        double expectedAve = -0.9286848221651098;

        assertEquals(expectedDim1,xgbEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,xgbEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,xgbEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,xgbEval.averageR2(),1e-6);
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        testXGBoost(p);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testXGBoost(p);
    }

    @Test
    public void testInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
            Model<Regressor> m = t.train(p.getA());
            m.predict(RegressionDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
            Model<Regressor> m = t.train(p.getA());
            m.predict(RegressionDataGenerator.emptyExample());
        });
    }
}
