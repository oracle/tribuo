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

package org.tribuo.regression.libsvm;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.common.libsvm.SVMParameters;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.libsvm.SVMRegressionType.SVMMode;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestLibSVM {

    private static final SVMParameters<Regressor> linearParams = new SVMParameters<>(new SVMRegressionType(SVMMode.EPSILON_SVR), KernelType.LINEAR);
    private static final LibSVMRegressionTrainer linear = new LibSVMRegressionTrainer(linearParams);
    private static final SVMParameters<Regressor> rbfParams = new SVMParameters<>(new SVMRegressionType(SVMMode.NU_SVR), KernelType.RBF);
    private static final LibSVMRegressionTrainer rbf = new LibSVMRegressionTrainer(rbfParams);
    private static final RegressionEvaluator eval = new RegressionEvaluator();

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(LibSVMTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    public static Model<Regressor> testLibSVM(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> linearModel = linear.train(p.getA());
        RegressionEvaluation linearEval = eval.evaluate(linearModel,p.getB());
        Model<Regressor> rbfModel = rbf.train(p.getA());
        RegressionEvaluation rbfEval = eval.evaluate(rbfModel,p.getB());
        return rbfModel;
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        Model<Regressor> model = testLibSVM(p);
        Helpers.testModelSerialization(model,Regressor.class);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testLibSVM(p);
    }

    @Test
    public void testInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
            Model<Regressor> m = linear.train(p.getA());
            m.predict(RegressionDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
            Model<Regressor> m = linear.train(p.getA());
            m.predict(RegressionDataGenerator.emptyExample());
        });
    }

    @Test
    public void testMultiDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
        testLibSVM(p);
    }

    @Test
    public void testThreeDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0);
        Model<Regressor> rbfModel = rbf.train(p.getA());
        RegressionEvaluation rbfEval = eval.evaluate(rbfModel,p.getB());
        double expectedDim1 = 0.041236330466452364;
        double expectedDim2 = 0.041236330466452364;
        double expectedDim3 = -0.02183233692613551;
        double expectedAve = 0.02021344133558974;

        assertEquals(expectedDim1,rbfEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,rbfEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,rbfEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,rbfEval.averageR2(),1e-6);
    }

    @Test
    public void testMultiSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();
        testLibSVM(p);
    }

    @Test
    public void testMultiInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
            Model<Regressor> m = linear.train(p.getA());
            m.predict(RegressionDataGenerator.invalidMultiDimSparseExample());
        });
    }

    @Test
    public void testMultiEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
            Model<Regressor> m = linear.train(p.getA());
            m.predict(RegressionDataGenerator.emptyMultiDimExample());
        });
    }
}
