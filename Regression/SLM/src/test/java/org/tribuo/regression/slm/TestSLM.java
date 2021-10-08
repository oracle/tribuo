/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression.slm;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.SparseModel;
import org.tribuo.Trainer;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.interop.onnx.DenseTransformer;
import org.tribuo.interop.onnx.ONNXExternalModel;
import org.tribuo.interop.onnx.RegressorTransformer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.NonlinearGaussianDataSource;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class TestSLM {
    private static final Logger logger = Logger.getLogger(TestSLM.class.getName());

    private static final SLMTrainer SFS = new SLMTrainer(false,-1);
    private static final SLMTrainer SFSN = new SLMTrainer(false,-1);
    private static final ElasticNetCDTrainer ELASTIC_NET = new ElasticNetCDTrainer(1.0,0.5,1e-4,500,false,0);
    private static final LARSTrainer LARS = new LARSTrainer(10);
    private static final LARSLassoTrainer LARS_LASSO = new LARSLassoTrainer(-1);

    private static final RegressionEvaluator e = new RegressionEvaluator();

    private static final URL TEST_REGRESSION_REORDER_ENET_MODEL = TestSLM.class.getResource("enet-4.1.0.model");
    private static final URL TEST_REGRESSION_REORDER_LARS_MODEL = TestSLM.class.getResource("lars-4.1.0.model");

    @BeforeAll
    public static void turnDownLogging() {
        Logger logger = Logger.getLogger(SLMTrainer.class.getName());
        logger.setLevel(Level.WARNING);
        logger = Logger.getLogger(ElasticNetCDTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    // This is a bit contrived, but it makes the trainer that failed appear in the stack trace.
    public static Model<Regressor> testTrainer(Trainer<Regressor> trainer,
                                               Pair<Dataset<Regressor>,Dataset<Regressor>> p,
                                               boolean testONNX) {
        Model<Regressor> m = trainer.train(p.getA());
        RegressionEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        if (testONNX) {
            try {
                SparseLinearModel slm = (SparseLinearModel) m;
                // Write out model
                Path onnxFile = Files.createTempFile("tribuo-slm-test", ".onnx");
                slm.saveONNXModel("org.tribuo.classification.sgd.linear.test", 1, onnxFile);

                // Prep mappings
                Map<String, Integer> featureMapping = new HashMap<>();
                for (VariableInfo f : slm.getFeatureIDMap()) {
                    VariableIDInfo id = (VariableIDInfo) f;
                    featureMapping.put(id.getName(), id.getID());
                }
                Map<Regressor, Integer> outputMapping = new HashMap<>();
                for (Pair<Integer, Regressor> l : slm.getOutputIDInfo()) {
                    outputMapping.put(l.getB(), l.getA());
                }

                String arch = System.getProperty("os.arch");
                if (arch.equalsIgnoreCase("amd64") || arch.equalsIgnoreCase("x86_64")) {
                    // Initialise the OrtEnvironment to load the native library
                    // (as OrtSession.SessionOptions doesn't trigger the static initializer).
                    OrtEnvironment env = OrtEnvironment.getEnvironment();
                    env.close();
                    // Load in via ORT
                    ONNXExternalModel<Regressor> onnxModel = ONNXExternalModel.createOnnxModel(new RegressionFactory(), featureMapping, outputMapping, new DenseTransformer(), new RegressorTransformer(), new OrtSession.SessionOptions(), onnxFile, "input");

                    // Generate predictions
                    List<Prediction<Regressor>> nativePredictions = slm.predict(p.getB());
                    List<Prediction<Regressor>> onnxPredictions = onnxModel.predict(p.getB());

                    // Assert the predictions are identical
                    for (int i = 0; i < nativePredictions.size(); i++) {
                        Prediction<Regressor> tribuo = nativePredictions.get(i);
                        Prediction<Regressor> external = onnxPredictions.get(i);
                        assertArrayEquals(tribuo.getOutput().getNames(), external.getOutput().getNames());
                        assertArrayEquals(tribuo.getOutput().getValues(), external.getOutput().getValues(), 1e-5);
                    }

                    // Check that the provenance can be extracted and is the same
                    ModelProvenance modelProv = m.getProvenance();
                    Optional<ModelProvenance> optProv = onnxModel.getTribuoProvenance();
                    assertTrue(optProv.isPresent());
                    ModelProvenance onnxProv = optProv.get();
                    assertNotSame(onnxProv, modelProv);
                    assertEquals(modelProv,onnxProv);

                    onnxModel.close();
                } else {
                    logger.warning("ORT based tests only supported on x86_64, found " + arch);
                }

                onnxFile.toFile().delete();
            } catch (IOException | OrtException ex) {
                fail(ex);
            }
        }
        return m;
    }

    public static Model<Regressor> testSFS(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testONNX) {
        return testTrainer(SFS,p,testONNX);
    }

    public static Model<Regressor> testSFSN(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testONNX) {
        return testTrainer(SFSN,p,testONNX);
    }

    public static Model<Regressor> testLARS(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testONNX) {
        return testTrainer(LARS,p,testONNX);
    }

    public static Model<Regressor> testLASSO(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testONNX) {
        return testTrainer(LARS_LASSO,p,testONNX);
    }

    public static Model<Regressor> testElasticNet(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testONNX) {
        return testTrainer(ELASTIC_NET,p,testONNX);
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        Model<Regressor> sfs = testSFS(p,false);
        Helpers.testModelSerialization(sfs,Regressor.class);
        Model<Regressor> sfsn = testSFSN(p,false);
        Helpers.testModelSerialization(sfsn,Regressor.class);
        Model<Regressor> lars = testLARS(p,false);
        Helpers.testModelSerialization(lars,Regressor.class);
        Model<Regressor> lasso = testLASSO(p,false);
        Helpers.testModelSerialization(lasso,Regressor.class);
        Model<Regressor> elastic = testElasticNet(p,false);
        Helpers.testModelSerialization(elastic,Regressor.class);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testSFS(p,false);
        testSFSN(p,false);
        testLARS(p,false);
        testLASSO(p,false);
        testElasticNet(p,false);
    }

    @Test
    public void testMultiDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
        testSFS(p,true);
        testSFSN(p,true);
        testLARS(p,true);
        testLASSO(p,true);
        testElasticNet(p,true);
    }

    @Test
    public void testMultiSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();
        testSFS(p,false);
        testSFSN(p,false);
        testLARS(p,false);
        testLASSO(p,false);
        testElasticNet(p,false);
    }

    @Test
    public void testNonlinearData() {
        DataSource<Regressor> trainSource = new NonlinearGaussianDataSource(200,new float[]{1.0f,2.0f,-3.0f,4.0f},1.0f,0.1f,-5.0f,5.0f,-1.0f,1.0f,42);
        DataSource<Regressor> testSource = new NonlinearGaussianDataSource(200,new float[]{1.0f,2.0f,-3.0f,4.0f},1.0f,0.1f,-5.0f,5.0f,-1.0f,1.0f,42*42);
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = new Pair<>(new MutableDataset<>(trainSource),new MutableDataset<>(testSource));
        testSFS(p,true);
        testSFSN(p,true);
        testLARS(p,true);
        Model<Regressor> lasso = testLASSO(p,true);
        Helpers.testModelSerialization(lasso,Regressor.class);
        testElasticNet(p,true);
    }

    @Test
    public void testThreeDenseDataLARS() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
        SparseModel<Regressor> llModel = LARS.train(p.getA());
        RegressionEvaluation llEval = e.evaluate(llModel,p.getB());
        double expectedDim1 = 0.5671244360433836;
        double expectedDim2 = 0.5671244360433927;
        double expectedDim3 = -2.457128076868633;
        double expectedAve = -0.44095973492728563;

        assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,llEval.averageR2(),1e-6);

        p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, true);
        SparseModel<Regressor> reorderedModel = LARS.train(p.getA());
        RegressionEvaluation reorderedEval = e.evaluate(reorderedModel,p.getB());

        assertEquals(expectedDim1,reorderedEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,reorderedEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,reorderedEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,reorderedEval.averageR2(),1e-6);
    }

    @Test
    public void testThreeDenseDataENet() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
        SparseModel<Regressor> llModel = ELASTIC_NET.train(p.getA());
        RegressionEvaluation llEval = e.evaluate(llModel,p.getB());
        double expectedDim1 = 0.5902193395184064;
        double expectedDim2 = 0.5902193395184064;
        double expectedDim3 = 0.2563468291737646;
        double expectedAve = 0.4789285027368592;

        assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,llEval.averageR2(),1e-6);

        p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, true);
        SparseModel<Regressor> reorderedModel = ELASTIC_NET.train(p.getA());
        RegressionEvaluation reorderedEval = e.evaluate(reorderedModel,p.getB());

        assertEquals(expectedDim1,reorderedEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,reorderedEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,reorderedEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,reorderedEval.averageR2(),1e-6);
    }

    @Test
    public void testRegressionReordering() throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(TEST_REGRESSION_REORDER_LARS_MODEL.openStream())) {
            @SuppressWarnings("unchecked")
            Model<Regressor> serializedModel = (Model<Regressor>) ois.readObject();
            Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
            RegressionEvaluation llEval = e.evaluate(serializedModel,p.getB());
            double expectedDim1 = 0.5671244360433836;
            double expectedDim2 = 0.5671244360433927;
            double expectedDim3 = -2.457128076868633;
            double expectedAve = -0.44095973492728563;

            assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedAve,llEval.averageR2(),1e-6);
        }
        try (ObjectInputStream ois = new ObjectInputStream(TEST_REGRESSION_REORDER_ENET_MODEL.openStream())) {
            @SuppressWarnings("unchecked")
            Model<Regressor> serializedModel = (Model<Regressor>) ois.readObject();
            Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
            RegressionEvaluation llEval = e.evaluate(serializedModel,p.getB());
            double expectedDim1 = 0.5902193395184064;
            double expectedDim2 = 0.5902193395184064;
            double expectedDim3 = 0.2563468291737646;
            double expectedAve = 0.4789285027368592;

            assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedAve,llEval.averageR2(),1e-6);
        }
    }
}
