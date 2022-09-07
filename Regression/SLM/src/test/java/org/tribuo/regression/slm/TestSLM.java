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

package org.tribuo.regression.slm;

import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.SparseModel;
import org.tribuo.SparseTrainer;
import org.tribuo.interop.onnx.OnnxTestUtils;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.NonlinearGaussianDataSource;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

public class TestSLM {
    private static final Logger logger = Logger.getLogger(TestSLM.class.getName());

    private static final SLMTrainer SFS = new SLMTrainer(false,-1);
    private static final SLMTrainer SFSN = new SLMTrainer(true,-1);
    private static final LARSTrainer LARS = new LARSTrainer(10);
    private static final LARSLassoTrainer LARS_LASSO = new LARSLassoTrainer(-1);

    private static final ElasticNetCDTrainer ELASTIC_NET = new ElasticNetCDTrainer(1.0,0.5,1e-4,500,false,0);
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
    public static SparseModel<Regressor> testTrainer(SparseTrainer<Regressor> trainer,
                                                     Pair<Dataset<Regressor>,Dataset<Regressor>> p,
                                                     boolean testONNX) {
        SparseModel<Regressor> m = trainer.train(p.getA());
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

                OnnxTestUtils.onnxRegressorComparison(slm,onnxFile,p.getB(),1e-4);

                onnxFile.toFile().delete();
            } catch (IOException | OrtException ex) {
                fail(ex);
            }
        }
        return m;
    }

    public static Model<Regressor> testSFS(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testONNX) {
        SparseModel<Regressor> newM = testTrainer(SFS,p,testONNX);
        return newM;
    }

    public static Model<Regressor> testSFSN(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testONNX) {
        SparseModel<Regressor> newM = testTrainer(SFSN,p,testONNX);
        return newM;
    }

    public static Model<Regressor> testLARS(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testONNX) {
        SparseModel<Regressor> newM = testTrainer(LARS,p,testONNX);
        return newM;
    }

    public static Model<Regressor> testLASSO(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testONNX) {
        SparseModel<Regressor> newM = testTrainer(LARS_LASSO,p,testONNX);
        return newM;
    }

    public static Model<Regressor> testElasticNet(Pair<Dataset<Regressor>,Dataset<Regressor>> p, boolean testONNX) {
        return testTrainer(ELASTIC_NET,p,testONNX);
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        Model<Regressor> sfs = testSFS(p,false);
        Helpers.testModelSerialization(sfs,Regressor.class);
        Helpers.testModelProtoSerialization(sfs,Regressor.class,p.getB());
        Model<Regressor> sfsn = testSFSN(p,false);
        Helpers.testModelSerialization(sfsn,Regressor.class);
        Helpers.testModelProtoSerialization(sfsn,Regressor.class,p.getB());
        Model<Regressor> lars = testLARS(p,false);
        Helpers.testModelSerialization(lars,Regressor.class);
        Helpers.testModelProtoSerialization(lars,Regressor.class,p.getB());
        Model<Regressor> lasso = testLASSO(p,false);
        Helpers.testModelSerialization(lasso,Regressor.class);
        Helpers.testModelProtoSerialization(lasso,Regressor.class,p.getB());
        Model<Regressor> elastic = testElasticNet(p,false);
        Helpers.testModelSerialization(elastic,Regressor.class);
        Helpers.testModelProtoSerialization(elastic,Regressor.class,p.getB());
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
    public void testSetInvocationCount() {
        // Create new trainer and dataset so as not to mess with the other tests
        ElasticNetCDTrainer originalTrainer = new ElasticNetCDTrainer(1.0,0.5);
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();

        // The number of times to call train before final training.
        // Original trainer will be trained numOfInvocations + 1 times
        // New trainer will have its invocation count set to numOfInvocations then trained once
        int numOfInvocations = 2;

        // Create the first model and train it numOfInvocations + 1 times
        SparseModel<Regressor> originalModel = null;
        for(int invocationCounter = 0; invocationCounter < numOfInvocations + 1; invocationCounter++){
            originalModel = originalTrainer.train(p.getA());
        }

        // Create a new model with same configuration, but set the invocation count to numOfInvocations
        // Assert that this succeeded, this means RNG will be at state where originalTrainer was before
        // it performed its last train.
        ElasticNetCDTrainer newTrainer = new ElasticNetCDTrainer(1.0,0.5);
        newTrainer.setInvocationCount(numOfInvocations);
        assertEquals(numOfInvocations,newTrainer.getInvocationCount());

        // Training newTrainer should now have the same result as if it
        // had trained numOfInvocations times previously even though it hasn't
        SparseModel<Regressor> newModel = newTrainer.train(p.getA());
        assertEquals(originalTrainer.getInvocationCount(),newTrainer.getInvocationCount());
    }

    @Test
    public void testNegativeInvocationCount(){
        assertThrows(IllegalArgumentException.class, () -> {
            ElasticNetCDTrainer t = new ElasticNetCDTrainer(1.0,0.5);
            t.setInvocationCount(-1);
        });
    }

    @Test
    public void testThreeDenseDataLARS() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
        SparseModel<Regressor> llModel = LARS.train(p.getA());
        Helpers.testModelProtoSerialization(llModel, Regressor.class, p.getB());
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
        Helpers.testModelProtoSerialization(llModel, Regressor.class, p.getB());
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
