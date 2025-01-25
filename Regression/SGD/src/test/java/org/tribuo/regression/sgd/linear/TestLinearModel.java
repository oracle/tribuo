/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression.sgd.linear;

import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.common.sgd.AbstractLinearTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.interop.onnx.OnnxTestUtils;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.optimisers.LBFGS;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.sgd.objectives.SquaredLoss;
import org.tribuo.test.Helpers;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestLinearModel {

    private static final LinearTrainer linear = new LinearTrainer(new SquaredLoss(), 50, false, 1e-4, 1e-4, 0);
    private static final RegressionEvaluator e = new RegressionEvaluator();

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{AbstractLinearTrainer.class, LinearTrainer.class, LBFGS.class};
        for (Class<?> c : classes) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
        }
    }

    //@Test
    public void testWine() throws IOException {
        var outputFactory = new RegressionFactory();
        var obj = new SquaredLoss();
        var linear = new LinearTrainer(obj, 100, false, 1e-4, 1e-4, 0);

        var csvLoader = new CSVLoader<>(';',outputFactory);
        var wineSource = csvLoader.loadDataSource(Paths.get("../../tutorials/winequality-red.csv"),"quality");
        var splitter = new TrainTestSplitter<>(wineSource, 0.7f, 0L);
        Dataset<Regressor> trainData = new MutableDataset<>(splitter.getTrain());
        Dataset<Regressor> testData = new MutableDataset<>(splitter.getTest());

        var evaluator = new RegressionEvaluator();
        System.out.printf("Training data size = %d, number of features = %d%n",trainData.size(),trainData.getFeatureMap().size());
        System.out.printf("Testing data size = %d, number of features = %d%n",testData.size(),testData.getFeatureMap().size());

        var lrStartTime = System.currentTimeMillis();
        var model = linear.train(trainData);
        var lrEndTime = System.currentTimeMillis();
        System.out.println("Training linear regression with l-bfgs took " + Util.formatDuration(lrStartTime,lrEndTime));

        var evaluation = evaluator.evaluate(model, trainData);
        System.out.println(evaluation.toString());

        var testEvaluation = evaluator.evaluate(model, testData);
        System.out.println(testEvaluation.toString());
    }

    public static Model<Regressor> testLinear(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = linear.train(p.getA());
        RegressionEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        return m;
    }

    @Test
    public void testOnnxSerialization() throws IOException, OrtException {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        LinearSGDModel model = linear.train(p.getA());

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-sgd-test",".onnx");
        model.saveONNXModel("org.tribuo.regression.sgd.linear.test",1,onnxFile);

        OnnxTestUtils.onnxRegressorComparison(model,onnxFile,p.getB(),1e-5);

        onnxFile.toFile().delete();
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        Model<Regressor> model = testLinear(p);
        Helpers.testModelProtoSerialization(model, Regressor.class, p.getB());
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testLinear(p);
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

    public static void testIndependentMultipleMultipleRegression(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        AbstractLinearSGDModel<Regressor> m = linear.train(p.getA());
        RegressionEvaluator e = new RegressionEvaluator();
        RegressionEvaluation evaluation = e.evaluate(m,p.getB());

        // The regressed values for the two dimensions are opposites, so the regression weights should
        // be the negatives of each other
        DenseMatrix weights = m.getWeightsCopy();
        DenseVector firstWeights = weights.getRow(0);
        DenseVector secondWeights = weights.getRow(1);
        secondWeights.scaleInPlace(-1.0);
        assertEquals(firstWeights,secondWeights);
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
    }

    @Test
    public void testMultiDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
        testIndependentMultipleMultipleRegression(p);
    }

    @Test
    public void testMultiSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();
        testIndependentMultipleMultipleRegression(p);
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
