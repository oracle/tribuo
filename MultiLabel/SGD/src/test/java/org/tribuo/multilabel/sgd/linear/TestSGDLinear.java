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

package org.tribuo.multilabel.sgd.linear;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.common.sgd.AbstractLinearSGDTrainer;
import org.tribuo.common.sgd.AbstractSGDTrainer;
import org.tribuo.interop.onnx.DenseTransformer;
import org.tribuo.interop.onnx.MultiLabelTransformer;
import org.tribuo.interop.onnx.ONNXExternalModel;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.multilabel.evaluation.MultiLabelEvaluation;
import org.tribuo.multilabel.example.MultiLabelDataGenerator;
import org.tribuo.multilabel.sgd.objectives.Hinge;
import org.tribuo.multilabel.sgd.objectives.BinaryCrossEntropy;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class TestSGDLinear {
    private static final Logger logger = Logger.getLogger(TestSGDLinear.class.getName());

    private static final LinearSGDTrainer hinge = new LinearSGDTrainer(new Hinge(),new AdaGrad(0.1,0.1),5,1000, Trainer.DEFAULT_SEED);
    private static final LinearSGDTrainer sigmoid = new LinearSGDTrainer(new BinaryCrossEntropy(),new AdaGrad(0.1,0.1),5,1000, Trainer.DEFAULT_SEED);

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{AbstractSGDTrainer.class, AbstractLinearSGDTrainer.class,LinearSGDTrainer.class};
        for (Class<?> c : classes) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
        }
    }

    @Test
    public void testPredictions() {
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();
        Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();

        testTrainer(train,test,hinge);
        testTrainer(train,test,sigmoid);
    }

    private static void testTrainer(Dataset<MultiLabel> train, Dataset<MultiLabel> test, LinearSGDTrainer trainer) {
        AbstractLinearSGDModel<MultiLabel> model = trainer.train(train);

        List<Prediction<MultiLabel>> predictions = model.predict(test);
        Prediction<MultiLabel> first = predictions.get(0);
        MultiLabel trueLabel = train.getOutputFactory().generateOutput("MONKEY,PUZZLE,TREE");
        assertEquals(trueLabel, first.getOutput(), "Predicted labels not equal");
        Map<String, List<Pair<String, Double>>> features = model.getTopFeatures(2);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());

        MultiLabelEvaluation evaluation = (MultiLabelEvaluation) train.getOutputFactory().getEvaluator().evaluate(model,test);

        Assertions.assertEquals(1.0, evaluation.microAveragedRecall());

        Helpers.testModelSerialization(model, MultiLabel.class);
    }

    @Test
    public void testOnnxSerialization() throws IOException, OrtException {
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();
        Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();
        LinearSGDModel model = (LinearSGDModel) sigmoid.train(train);

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-sgd-test",".onnx");
        model.saveONNXModel("org.tribuo.multilabel.sgd.linear.test",1,onnxFile);

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
        if (arch.equalsIgnoreCase("amd64") || arch.equalsIgnoreCase("x86_64")) {
            // Initialise the OrtEnvironment to load the native library
            // (as OrtSession.SessionOptions doesn't trigger the static initializer).
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            env.close();
            // Load in via ORT
            ONNXExternalModel<MultiLabel> onnxModel = ONNXExternalModel.createOnnxModel(new MultiLabelFactory(),featureMapping,outputMapping,new DenseTransformer(),new MultiLabelTransformer(),new OrtSession.SessionOptions(),onnxFile,"input");

            // Generate predictions
            List<Prediction<MultiLabel>> nativePredictions = model.predict(test);
            List<Prediction<MultiLabel>> onnxPredictions = onnxModel.predict(test);

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
                        assertEquals(l.getValue().getScore(),other.getScore(),1e-6);
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
}
