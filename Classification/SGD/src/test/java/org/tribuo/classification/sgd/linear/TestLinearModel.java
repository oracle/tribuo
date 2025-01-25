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

package org.tribuo.classification.sgd.linear;

import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.common.sgd.AbstractLinearTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.dataset.DatasetView;
import org.tribuo.datasource.IDXDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.interop.onnx.OnnxTestUtils;
import org.tribuo.math.optimisers.LBFGS;
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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestLinearModel {
    private static final LinearTrainer logistic = new LinearTrainer(new LogMulticlass(), 50, false, 1e-4, 1e-4, 0);
    private static final LinearTrainer hinge = new LinearTrainer(new Hinge(), 50, false, 1e-4, 1e-4, 0);

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{AbstractLinearTrainer.class, LinearTrainer.class, LBFGS.class};
        for (Class<?> c : classes) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
        }
    }

    //@Test
    public void testMNIST() throws IOException {
        var outputFactory = new LabelFactory();
        var obj = new LogMulticlass();
        var linear = new LinearTrainer(obj, 50, false, 1e-4, 1e-4, 0);
        var mnistSource = new IDXDataSource<>(Path.of("../../tutorials/train-images-idx3-ubyte.gz"), Path.of("../..//tutorials/train-labels-idx1-ubyte.gz"), outputFactory);
        var testSource = new IDXDataSource<>(Path.of("../../tutorials/t10k-images-idx3-ubyte.gz"), Path.of("../..//tutorials/t10k-labels-idx1-ubyte.gz"), outputFactory);

        var fullTrainData = new MutableDataset<>(mnistSource);
        var trainData = DatasetView.createBootstrapView(fullTrainData, 6000, 12345);
        var testData = new MutableDataset<>(testSource);

        var evaluator = new LabelEvaluator();
        System.out.printf("Training data size = %d, number of features = %d, number of classes = %d%n",trainData.size(),trainData.getFeatureMap().size(),trainData.getOutputInfo().size());
        System.out.printf("Testing data size = %d, number of features = %d, number of classes = %d%n",testData.size(),testData.getFeatureMap().size(),testData.getOutputInfo().size());

        var lrStartTime = System.currentTimeMillis();
        var model = linear.train(trainData);
        var lrEndTime = System.currentTimeMillis();
        System.out.println("Training logistic regression with l-bfgs took " + Util.formatDuration(lrStartTime,lrEndTime));

        var evaluation = evaluator.evaluate(model, trainData);
        System.out.println(evaluation.toString());

        var testEvaluation = evaluator.evaluate(model, testData);
        System.out.println(testEvaluation.toString());
    }

    //@Test
    public void testIrises() throws IOException {
        var outputFactory = new LabelFactory();
        var obj = new LogMulticlass();
        var linear = new LinearTrainer(obj, 50, false, 1e-4, 1e-4, 0);
        var csvLoader = new CSVLoader<>(outputFactory);

        var irisHeaders = new String[]{"sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"};
        var irisesSource = csvLoader.loadDataSource(Paths.get("../..//tutorials/bezdekIris.data"),"species",irisHeaders);
        var irisSplitter = new TrainTestSplitter<>(irisesSource,0.7,1L);

        var trainData = new MutableDataset<>(irisSplitter.getTrain());
        var testData = new MutableDataset<>(irisSplitter.getTest());

        System.out.printf("Training data size = %d, number of features = %d, number of classes = %d%n",trainData.size(),trainData.getFeatureMap().size(),trainData.getOutputInfo().size());
        System.out.printf("Testing data size = %d, number of features = %d, number of classes = %d%n",testData.size(),testData.getFeatureMap().size(),testData.getOutputInfo().size());

        var evaluator = new LabelEvaluator();
        var lrStartTime = System.currentTimeMillis();
        var model = linear.train(trainData);
        var lrEndTime = System.currentTimeMillis();
        System.out.println("Training logistic regression with l-bfgs took " + Util.formatDuration(lrStartTime,lrEndTime));

        var evaluation = evaluator.evaluate(model, trainData);
        System.out.println(evaluation.toString());

        var testEvaluation = evaluator.evaluate(model, testData);
        System.out.println(testEvaluation.toString());
    }

    public static LinearSGDModel testLinear(Pair<Dataset<Label>,Dataset<Label>> p, LinearTrainer trainer) {
        LinearSGDModel m = trainer.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        assertFalse(features.isEmpty());
        return m;
    }

    @Test
    public void testSingleClassTraining() {
        Pair<Dataset<Label>,Dataset<Label>> data = LabelledDataGenerator.denseTrainTest();
        DatasetView<Label> trainingData = DatasetView.createView(data.getA(),(Example<Label> e) -> e.getOutput().getLabel().equals("Foo"), "Foo selector");
        Model<Label> model = logistic.train(trainingData);
        LabelEvaluation evaluation = (LabelEvaluation) trainingData.getOutputFactory().getEvaluator().evaluate(model,data.getB());
        assertEquals(0.0,evaluation.accuracy(new Label("Baz")));
        assertEquals(1.0,evaluation.accuracy(new Label("Quux")));
        assertEquals(1.0,evaluation.recall(new Label("Foo")));
        assertEquals(0.0,evaluation.recall(new Label("Bar")));
    }

    @Test
    public void testOnnxSerialization() throws IOException, OrtException {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        LinearSGDModel model = logistic.train(p.getA());

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-sgd-test",".onnx");
        model.saveONNXModel("org.tribuo.classification.sgd.linear.test",1,onnxFile);

        OnnxTestUtils.onnxLabelComparison(model,onnxFile,p.getB(),1e-6);

        onnxFile.toFile().delete();
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        Model<Label> model = testLinear(p, logistic);
        Helpers.testModelProtoSerialization(model, Label.class, p.getB());
        model = testLinear(p, hinge);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testLinear(p, logistic);
    }

    @Test
    public void testSparseBinaryData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testLinear(p, logistic);
    }

    @Test
    public void testInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = logistic.train(p.getA());
            m.predict(LabelledDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = logistic.train(p.getA());
            m.predict(LabelledDataGenerator.emptyExample());
        });
    }

}
