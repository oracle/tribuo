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

package org.tribuo.classification.sgd.linear;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.common.sgd.AbstractLinearSGDTrainer;
import org.tribuo.common.sgd.AbstractSGDTrainer;
import org.tribuo.dataset.DatasetView;
import org.tribuo.interop.onnx.DenseTransformer;
import org.tribuo.interop.onnx.LabelTransformer;
import org.tribuo.interop.onnx.ONNXExternalModel;
import org.tribuo.math.optimisers.AdaGrad;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class TestSGDLinear {
    private static final Logger logger = Logger.getLogger(TestSGDLinear.class.getName());

    private static final LinearSGDTrainer t = new LinearSGDTrainer(new LogMulticlass(),new AdaGrad(0.1,0.1),5,1000, Trainer.DEFAULT_SEED);

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{AbstractSGDTrainer.class, AbstractLinearSGDTrainer.class, LinearSGDTrainer.class};
        for (Class<?> c : classes) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
        }
    }

    public static LinearSGDModel testSGDLinear(Pair<Dataset<Label>,Dataset<Label>> p) {
        LinearSGDModel m = (LinearSGDModel) t.train(p.getA());
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
        // Done to reduce sensitivity to test run order.
        Trainer<Label> trainer = new LinearSGDTrainer(new Hinge(),new AdaGrad(0.1,0.1),5,1000, Trainer.DEFAULT_SEED);
        DatasetView<Label> trainingData = DatasetView.createView(data.getA(),(Example<Label> e) -> e.getOutput().getLabel().equals("Foo"), "Foo selector");
        Model<Label> model = trainer.train(trainingData);
        LabelEvaluation evaluation = (LabelEvaluation) trainingData.getOutputFactory().getEvaluator().evaluate(model,data.getB());
        assertEquals(0.0,evaluation.accuracy(new Label("Baz")));
        assertEquals(0.0,evaluation.accuracy(new Label("Quux")));
        assertEquals(1.0,evaluation.recall(new Label("Foo")));
        assertEquals(1.0,evaluation.recall(new Label("Bar"))); // This is due to the random init of the classifier.
    }

    @Test
    public void testOnnxSerialization() throws IOException, OrtException {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        LinearSGDModel model = (LinearSGDModel) t.train(p.getA());

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-sgd-test",".onnx");
        model.saveONNXModel("org.tribuo.classification.sgd.linear.test",1,onnxFile);

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
        if (arch.equalsIgnoreCase("amd64") || arch.equalsIgnoreCase("x86_64")) {
            // Initialise the OrtEnvironment to load the native library
            // (as OrtSession.SessionOptions doesn't trigger the static initializer).
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            env.close();
            // Load in via ORT
            ONNXExternalModel<Label> onnxModel = ONNXExternalModel.createOnnxModel(new LabelFactory(), featureMapping, outputMapping, new DenseTransformer(), new LabelTransformer(), new OrtSession.SessionOptions(), onnxFile, "input");

            // Generate predictions
            List<Prediction<Label>> nativePredictions = model.predict(p.getB());
            List<Prediction<Label>> onnxPredictions = onnxModel.predict(p.getB());

            // Assert the predictions are identical
            for (int i = 0; i < nativePredictions.size(); i++) {
                Prediction<Label> tribuo = nativePredictions.get(i);
                Prediction<Label> external = onnxPredictions.get(i);
                assertEquals(tribuo.getOutput().getLabel(), external.getOutput().getLabel());
                assertEquals(tribuo.getOutput().getScore(), external.getOutput().getScore(), 1e-6);
                for (Map.Entry<String, Label> l : tribuo.getOutputScores().entrySet()) {
                    Label other = external.getOutputScores().get(l.getKey());
                    if (other == null) {
                        fail("Failed to find label " + l.getKey() + " in ORT prediction.");
                    } else {
                        assertEquals(l.getValue().getScore(), other.getScore(), 1e-6);
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

        onnxFile.toFile().delete();
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        Model<Label> model = testSGDLinear(p);
        Helpers.testModelSerialization(model,Label.class);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testSGDLinear(p);
    }

    @Test
    public void testSparseBinaryData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testSGDLinear(p);
    }

    @Test
    public void testInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = t.train(p.getA());
            m.predict(LabelledDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = t.train(p.getA());
            m.predict(LabelledDataGenerator.emptyExample());
        });
    }

    @ParameterizedTest
    @ValueSource(strings = {"label-linear-sgd-4.0.2.model"})
    public void testSerializedModel(String resourceName) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(TestSGDLinear.class.getResource(resourceName).openStream())) {
           Model<?> model = (Model<?>) ois.readObject();
           if (model.validate(Label.class)) {
               @SuppressWarnings("unchecked") // Guarded by validate call.
               Model<Label> m = (Model<Label>) model;
               LabelEvaluator e = new LabelEvaluator();
               LabelEvaluation evaluation = e.evaluate(m,LabelledDataGenerator.denseTrainTest().getB());
               Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
               Assertions.assertNotNull(features);
               assertFalse(features.isEmpty());
               features = m.getTopFeatures(-1);
               Assertions.assertNotNull(features);
               assertFalse(features.isEmpty());
           } else {
               fail("Invalid model type found, expected Label");
           }
        }
    }
}
