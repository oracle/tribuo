/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Utilities for comparing onnx models to Tribuo models in the tests.
 */
public class OnnxTestUtils {
    private static final Logger logger = Logger.getLogger(OnnxTestUtils.class.getName());

    /**
     * Runs a comparison between an onnx model and a Tribuo model.
     * <p>
     * Only runs on x86 platforms.
     * @param model The Tribuo model.
     * @param onnxFile The path on disk to the ONNX model.
     * @param testSet The test set.
     * @param delta The delta allowable between the Tribuo and ORT predictions.
     * @throws OrtException If ORT failed to initialize.
     */
    public static void onnxLabelComparison(Model<Label> model, Path onnxFile, Dataset<Label> testSet, double delta) throws OrtException {
        // Prep mappings
        Map<String, Integer> featureMapping = new HashMap<>();
        for (VariableInfo f : model.getFeatureIDMap()){
            VariableIDInfo id = (VariableIDInfo) f;
            featureMapping.put(id.getName(),id.getID());
        }
        Map<Label, Integer> outputMapping = new HashMap<>();
        for (Pair<Integer,Label> l : model.getOutputIDInfo()) {
            outputMapping.put(l.getB(), l.getA());
        }

        String arch = System.getProperty("os.arch");
        String os = System.getProperty("os.name").toLowerCase(Locale.ENGLISH);
        if (arch.equalsIgnoreCase("amd64") || arch.equalsIgnoreCase("x86_64") || (arch.equalsIgnoreCase("aarch64") && (os.contains("mac") || os.contains("nux")))) {
            // Initialise the OrtEnvironment to load the native library
            // (as OrtSession.SessionOptions doesn't trigger the static initializer).
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            env.close();
            // Load in via ORT
            ONNXExternalModel<Label> onnxModel = ONNXExternalModel.createOnnxModel(new LabelFactory(), featureMapping, outputMapping, new DenseTransformer(), new LabelTransformer(), new OrtSession.SessionOptions(), onnxFile, "input");

            // Generate predictions
            List<Prediction<Label>> nativePredictions = model.predict(testSet);
            List<Prediction<Label>> onnxPredictions = onnxModel.predict(testSet);

            // Assert the predictions are identical
            for (int i = 0; i < nativePredictions.size(); i++) {
                Prediction<Label> tribuo = nativePredictions.get(i);
                Prediction<Label> external = onnxPredictions.get(i);
                assertEquals(tribuo.getOutput().getLabel(), external.getOutput().getLabel());
                assertEquals(tribuo.getOutput().getScore(), external.getOutput().getScore(), delta);
                for (Map.Entry<String, Label> l : tribuo.getOutputScores().entrySet()) {
                    Label other = external.getOutputScores().get(l.getKey());
                    if (other == null) {
                        fail("Failed to find label " + l.getKey() + " in ORT prediction.");
                    } else {
                        assertEquals(l.getValue().getScore(), other.getScore(), delta);
                    }
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

    }

    /**
     * Runs a comparison between an onnx model and a Tribuo model.
     * <p>
     * Only runs on x86 platforms.
     * @param model The Tribuo model.
     * @param onnxFile The path on disk to the ONNX model.
     * @param testSet The test set.
     * @param delta The delta allowable between the Tribuo and ORT predictions.
     * @throws OrtException If ORT failed to initialize.
     */
    public static void onnxMultiLabelComparison(Model<MultiLabel> model, Path onnxFile, Dataset<MultiLabel> testSet, double delta) throws OrtException {
        // Prep mappings
        Map<String, Integer> featureMapping = new HashMap<>();
        for (VariableInfo f : model.getFeatureIDMap()){
            VariableIDInfo id = (VariableIDInfo) f;
            featureMapping.put(id.getName(),id.getID());
        }
        Map<MultiLabel, Integer> outputMapping = new HashMap<>();
        for (Pair<Integer,MultiLabel> l : model.getOutputIDInfo()) {
            outputMapping.put(l.getB(), l.getA());
        }

        String arch = System.getProperty("os.arch");
        String os = System.getProperty("os.name").toLowerCase(Locale.ENGLISH);
        if (arch.equalsIgnoreCase("amd64") || arch.equalsIgnoreCase("x86_64") || (arch.equalsIgnoreCase("aarch64") && (os.contains("mac") || os.contains("nux")))) {
            // Initialise the OrtEnvironment to load the native library
            // (as OrtSession.SessionOptions doesn't trigger the static initializer).
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            env.close();
            // Load in via ORT
            ONNXExternalModel<MultiLabel> onnxModel = ONNXExternalModel.createOnnxModel(new MultiLabelFactory(),featureMapping,outputMapping,new DenseTransformer(),new MultiLabelTransformer(),new OrtSession.SessionOptions(),onnxFile,"input");

            // Generate predictions
            List<Prediction<MultiLabel>> nativePredictions = model.predict(testSet);
            List<Prediction<MultiLabel>> onnxPredictions = onnxModel.predict(testSet);

            // Assert the predictions are identical
            for (int i = 0; i < nativePredictions.size(); i++) {
                Prediction<MultiLabel> tribuo = nativePredictions.get(i);
                Prediction<MultiLabel> external = onnxPredictions.get(i);
                assertEquals(tribuo.getOutput().getLabelSet(), external.getOutput().getLabelSet());
                for (Map.Entry<String,MultiLabel> l : tribuo.getOutputScores().entrySet()) {
                    MultiLabel other = external.getOutputScores().get(l.getKey());
                    if (other == null) {
                        fail("Failed to find label " + l.getKey() + " in ORT prediction.");
                    } else {
                        assertEquals(l.getValue().getScore(),other.getScore(),delta);
                    }
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

    }

    /**
     * Runs a comparison between an onnx model and a Tribuo model.
     * <p>
     * Only runs on x86 platforms.
     * @param model The Tribuo model.
     * @param onnxFile The path on disk to the ONNX model.
     * @param testSet The test set.
     * @param delta The delta allowable between the Tribuo and ORT predictions.
     * @throws OrtException If ORT failed to initialize.
     */
    public static void onnxRegressorComparison(Model<Regressor> model, Path onnxFile, Dataset<Regressor> testSet, double delta) throws OrtException {
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
        String os = System.getProperty("os.name").toLowerCase(Locale.ENGLISH);
        if (arch.equalsIgnoreCase("amd64") || arch.equalsIgnoreCase("x86_64") || (arch.equalsIgnoreCase("aarch64") && (os.contains("mac") || os.contains("nux")))) {
            // Initialise the OrtEnvironment to load the native library
            // (as OrtSession.SessionOptions doesn't trigger the static initializer).
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            env.close();
            // Load in via ORT
            ONNXExternalModel<Regressor> onnxModel = ONNXExternalModel.createOnnxModel(new RegressionFactory(),featureMapping,outputMapping,new DenseTransformer(),new RegressorTransformer(),new OrtSession.SessionOptions(),onnxFile,"input");

            // Generate predictions
            List<Prediction<Regressor>> nativePredictions = model.predict(testSet);
            List<Prediction<Regressor>> onnxPredictions = onnxModel.predict(testSet);

            // Assert the predictions are identical
            for (int i = 0; i < nativePredictions.size(); i++) {
                Prediction<Regressor> tribuo = nativePredictions.get(i);
                Prediction<Regressor> external = onnxPredictions.get(i);
                assertArrayEquals(tribuo.getOutput().getNames(),external.getOutput().getNames());
                assertArrayEquals(tribuo.getOutput().getValues(),external.getOutput().getValues(),delta);
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
    }

}
