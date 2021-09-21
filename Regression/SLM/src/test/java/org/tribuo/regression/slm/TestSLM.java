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
import org.tribuo.SparseModel;
import org.tribuo.Trainer;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.interop.onnx.DenseTransformer;
import org.tribuo.interop.onnx.ONNXExternalModel;
import org.tribuo.interop.onnx.RegressorTransformer;
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.fail;

public class TestSLM {
    private static final Logger logger = Logger.getLogger(TestSLM.class.getName());

    private static final SLMTrainer SFS = new SLMTrainer(false,-1);
    private static final SLMTrainer SFSN = new SLMTrainer(false,-1);
    private static final ElasticNetCDTrainer ELASTIC_NET = new ElasticNetCDTrainer(1.0,0.5);
    private static final LARSTrainer LARS = new LARSTrainer(-1);
    private static final LARSLassoTrainer LARS_LASSO = new LARSLassoTrainer(-1);

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
        RegressionEvaluator e = new RegressionEvaluator();
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
    public void testSetInvocationCount() {
        // Create new trainer and dataset so as not to mess with the other tests
        ElasticNetCDTrainer originalTrainer = new ElasticNetCDTrainer(1.0,0.5);
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();

        // The number of times to call train before final training.
        // Original trainer will be trained numOfInvocations + 1 times
        // New trainer will have it's invocation count set to numOfInvocations then trained once
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

}
