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

package org.tribuo.regression.libsvm;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.oracle.labs.mlrg.olcut.util.Pair;
import libsvm.svm_model;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.common.libsvm.SVMParameters;
import org.tribuo.interop.onnx.DenseTransformer;
import org.tribuo.interop.onnx.ONNXExternalModel;
import org.tribuo.interop.onnx.RegressorTransformer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.libsvm.SVMRegressionType.SVMMode;
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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestLibSVM {
    private static final Logger logger = Logger.getLogger(TestLibSVM.class.getName());

    private static final SVMParameters<Regressor> linearParams = new SVMParameters<>(new SVMRegressionType(SVMMode.EPSILON_SVR), KernelType.LINEAR);
    private static final LibSVMRegressionTrainer linear = new LibSVMRegressionTrainer(linearParams);
    private static final SVMParameters<Regressor> rbfParams;
    private static final LibSVMRegressionTrainer rbf;
    private static final LibSVMRegressionTrainer linStandardize = new LibSVMRegressionTrainer(new SVMParameters<>(new SVMRegressionType(SVMMode.NU_SVR), KernelType.LINEAR),true);
    private static final RegressionEvaluator eval = new RegressionEvaluator();
    static {
        rbfParams = new SVMParameters<>(new SVMRegressionType(SVMMode.NU_SVR),KernelType.RBF);
        rbfParams.setGamma(0.5);
        rbfParams.setNu(0.5);
        rbfParams.setEpsilon(0.5);
        rbf = new LibSVMRegressionTrainer(rbfParams);
    }

    private static final URL TEST_REGRESSION_REORDER_MODEL = TestLibSVM.class.getResource("libsvm-4.1.0.model");

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(LibSVMTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    public static Model<Regressor> testLibSVM(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        LibSVMModel<Regressor> linearModel = linear.train(p.getA());
        RegressionEvaluation linearEval = eval.evaluate(linearModel,p.getB());
        LibSVMRegressionModel rbfModel = (LibSVMRegressionModel) rbf.train(p.getA());
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
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0,true);
        Model<Regressor> rbfModel = rbf.train(p.getA());
        RegressionEvaluation rbfEval = eval.evaluate(rbfModel,p.getB());
        double expectedDim1 = 0.0038608193481045605;
        double expectedDim2 = 0.0038608193481045605;
        double expectedDim3 = -0.12392916600305548;
        double expectedAve = -0.03873584243561545;

        assertEquals(expectedDim1,rbfEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,rbfEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,rbfEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,rbfEval.averageR2(),1e-6);
    }

    @Test
    public void testMultiModelsDifferent() {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();

        LibSVMRegressionModel rbfModel = (LibSVMRegressionModel) rbf.train(p.getA());
        List<svm_model> rbfModelList = rbfModel.getInnerModels();
        assertEquals(2, rbfModelList.size());
        double[] firstSV = rbfModelList.get(0).sv_coef[0];
        double[] secondSV = rbfModelList.get(1).sv_coef[0];
        // The two dimensions are the inverse of each other, and should have inverted sv_coef.
        for (int i = 0; i < firstSV.length; i++) {
            firstSV[i] = -firstSV[i];
        }
        assertArrayEquals(firstSV, secondSV);
        Helpers.testModelProtoSerialization(rbfModel, Regressor.class, p.getB());
    }

    @Test
    public void testMultiStandardizedModelsDifferent() {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();

        LibSVMRegressionModel linSModel = (LibSVMRegressionModel) linStandardize.train(p.getA());
        List<svm_model> linModelList = linSModel.getInnerModels();
        assertEquals(2,linModelList.size());
        double[] means = linSModel.getMeans();
        double[] variances = linSModel.getVariances();
        assertEquals(means[0],-means[1]);
        assertEquals(variances[0],variances[1]);
        Helpers.testModelProtoSerialization(linSModel, Regressor.class, p.getB());
        // The two dimensions are the inverse of each other, and should have inverted sv_coef.
        // However the fact that some values are negative means the sv_coefs end up slightly different,
        // and it appears to happen inside LibSVM.
        /*
        double[] firstSV = linModelList.get(0).sv_coef[0];
        double[] secondSV = linModelList.get(1).sv_coef[0];
        for (int i = 0; i < firstSV.length; i++) {
            firstSV[i] = -firstSV[i];
        }
        assertArrayEquals(firstSV,secondSV);
         */
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

    @Test
    public void testRegressionReordering() throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(TEST_REGRESSION_REORDER_MODEL.openStream())) {
            @SuppressWarnings("unchecked")
            Model<Regressor> serializedModel = (Model<Regressor>) ois.readObject();
            Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
            RegressionEvaluation llEval = eval.evaluate(serializedModel,p.getB());
            double expectedDim1 = 0.0038608193481045605;
            double expectedDim2 = 0.0038608193481045605;
            double expectedDim3 = -0.12392916600305548;
            double expectedAve = -0.03873584243561545;

            assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
            assertEquals(expectedAve,llEval.averageR2(),1e-6);
        }
    }


    @Test
    public void testOnnxSerialization() throws IOException, OrtException {
        Pair<Dataset<Regressor>,Dataset<Regressor>> dense = RegressionDataGenerator.denseTrainTest();

        testOnnxSerialization(dense,linear);
        testOnnxSerialization(dense,linStandardize);
        testOnnxSerialization(dense,rbf);

        Pair<Dataset<Regressor>,Dataset<Regressor>> threeDim = RegressionDataGenerator.threeDimDenseTrainTest(1.0,false);

        testOnnxSerialization(threeDim,linear);
        testOnnxSerialization(threeDim,linStandardize);
        testOnnxSerialization(threeDim,rbf);
    }

    private static void testOnnxSerialization(Pair<Dataset<Regressor>,Dataset<Regressor>> datasetPair, LibSVMRegressionTrainer trainer) throws IOException, OrtException {
        LibSVMRegressionModel model = (LibSVMRegressionModel) trainer.train(datasetPair.getA());

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-libsvm-test",".onnx");
        model.saveONNXModel("org.tribuo.regression.libsvm.test",1,onnxFile);

        // Prep mappings
        Map<String, Integer> featureMapping = new HashMap<>();
        for (VariableInfo f : model.getFeatureIDMap()){
            VariableIDInfo id = (VariableIDInfo) f;
            featureMapping.put(id.getName(),id.getID());
        }
        Map<Regressor, Integer> outputMapping = new HashMap<>();
        for (Pair<Integer,Regressor> l : model.getOutputIDInfo()) {
            outputMapping.put(l.getB(), l.getA());
        }

        String arch = System.getProperty("os.arch");
        if (arch.equalsIgnoreCase("amd64") || arch.equalsIgnoreCase("x86_64")) {
            // Initialise the OrtEnvironment to load the native library
            // (as OrtSession.SessionOptions doesn't trigger the static initializer).
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            env.close();
            // Load in via ORT
            ONNXExternalModel<Regressor> onnxModel = ONNXExternalModel.createOnnxModel(new RegressionFactory(),featureMapping,outputMapping,new DenseTransformer(),new RegressorTransformer(),new OrtSession.SessionOptions(),onnxFile,"input");

            // Generate predictions
            List<Prediction<Regressor>> nativePredictions = model.predict(datasetPair.getB());
            List<Prediction<Regressor>> onnxPredictions = onnxModel.predict(datasetPair.getB());

            // Assert the predictions are identical
            for (int i = 0; i < nativePredictions.size(); i++) {
                Prediction<Regressor> tribuo = nativePredictions.get(i);
                Prediction<Regressor> external = onnxPredictions.get(i);
                assertArrayEquals(tribuo.getOutput().getNames(),external.getOutput().getNames());
                if (model.isStandardized()) {
                    // Standardized models are less numerically stable when cast into floats
                    double[] tribuoValues = tribuo.getOutput().getValues();
                    double[] externalValues = external.getOutput().getValues();
                    assertEquals(tribuoValues.length,externalValues.length);
                    for (int j = 0; j < tribuoValues.length; j++) {
                        // compute sf comparison
                        assertEquals(tribuoValues[j], externalValues[j], Math.abs(tribuoValues[j])/1e3);
                    }
                } else {
                    assertArrayEquals(tribuo.getOutput().getValues(),external.getOutput().getValues(),1e-4);
                }
            }

            // Check that the provenance can be extracted and is the same
            ModelProvenance modelProv = model.getProvenance();
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
    }
}
