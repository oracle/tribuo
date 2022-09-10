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

package org.tribuo.regression.liblinear;

import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.common.liblinear.LibLinearModel;
import org.tribuo.common.liblinear.LibLinearTrainer;
import org.tribuo.interop.onnx.OnnxTestUtils;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.liblinear.LinearRegressionType.LinearType;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestLibLinear {
    private static final Logger logger = Logger.getLogger(TestLibLinear.class.getName());

    private static final LibLinearRegressionTrainer t = new LibLinearRegressionTrainer(new LinearRegressionType(LinearType.L2R_L2LOSS_SVR_DUAL),1.0,1000,0.1,0.5);
    private static final RegressionEvaluator e = new RegressionEvaluator();

    private static final URL TEST_REGRESSION_REORDER_MODEL = TestLibLinear.class.getResource("liblinear-4.1.0.model");

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(LibLinearTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    public static Model<Regressor> testLibLinear(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> linearModel = t.train(p.getA());
        RegressionEvaluation evaluation = e.evaluate(linearModel,p.getB());
        return linearModel;
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        Model<Regressor> model = testLibLinear(p);
        Helpers.testModelSerialization(model, Regressor.class);
        Helpers.testModelProtoSerialization(model, Regressor.class, p.getB());
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testLibLinear(p);
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

    @Test
    public void testMultiDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
        testLibLinear(p);
    }

    @Test
    public void testThreeDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
        LibLinearRegressionTrainer localTrainer = new LibLinearRegressionTrainer(new LinearRegressionType(LinearType.L2R_L2LOSS_SVR_DUAL),1.0,1000,0.1,0.5);
        localTrainer.forceZero = true;
        LibLinearModel<Regressor> llModel = localTrainer.train(p.getA());
        RegressionEvaluation llEval = e.evaluate(llModel,p.getB());
        double expectedDim1 = 0.6634367596601265;
        double expectedDim2 = 0.6634367596601265;
        double expectedDim3 = 0.01112107563226139;
        double expectedAve = 0.4459981983175048;

        assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,llEval.averageR2(),1e-6);

        p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, true);
        llModel = localTrainer.train(p.getA());
        llEval = e.evaluate(llModel,p.getB());

        assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,llEval.averageR2(),1e-6);
    }

    @Test
    public void testMultiSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();
        testLibLinear(p);
    }

    @Test
    public void testMultiInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
            Model<Regressor> m = t.train(p.getA());
            m.predict(RegressionDataGenerator.invalidMultiDimSparseExample());
        });
    }

    @Test
    public void testMultiEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
            Model<Regressor> m = t.train(p.getA());
            m.predict(RegressionDataGenerator.emptyMultiDimExample());
        });
    }

    @Test
    public void testRegressionReordering() throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(TEST_REGRESSION_REORDER_MODEL.openStream())) {
            @SuppressWarnings("unchecked")
            Model<Regressor> serializedModel = (Model<Regressor>) ois.readObject();
            Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
            RegressionEvaluation llEval = e.evaluate(serializedModel,p.getB());
            double expectedDim1 = 0.6634367596601265;
            double expectedDim2 = 0.6634367596601265;
            double expectedDim3 = 0.01112107563226139;
            double expectedAve = 0.4459981983175048;

            assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedAve,llEval.averageR2(),1e-6);
        }
    }

    @Test
    public void testOnnxSerialization() throws IOException, OrtException {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        LibLinearRegressionModel model = (LibLinearRegressionModel) t.train(p.getA());

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-liblinear-test",".onnx");
        model.saveONNXModel("org.tribuo.regression.liblinear.test",1,onnxFile);

        OnnxTestUtils.onnxRegressorComparison(model,onnxFile,p.getB(),1e-5);

        onnxFile.toFile().delete();
    }
}
