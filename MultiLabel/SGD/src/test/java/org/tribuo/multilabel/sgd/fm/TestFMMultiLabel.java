/*
 * Copyright (c) 2021, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.multilabel.sgd.fm;

import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.common.sgd.AbstractFMTrainer;
import org.tribuo.common.sgd.AbstractSGDTrainer;
import org.tribuo.interop.onnx.OnnxTestUtils;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.evaluation.MultiLabelEvaluation;
import org.tribuo.multilabel.example.MultiLabelDataGenerator;
import org.tribuo.multilabel.sgd.objectives.BinaryCrossEntropy;
import org.tribuo.multilabel.sgd.objectives.Hinge;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestFMMultiLabel {
    private static final Logger logger = Logger.getLogger(TestFMMultiLabel.class.getName());

    private static final FMMultiLabelTrainer hinge = new FMMultiLabelTrainer(new Hinge(),new AdaGrad(0.1,0.1),5,1000, Trainer.DEFAULT_SEED, 5,0.1);
    private static final FMMultiLabelTrainer sigmoid = new FMMultiLabelTrainer(new BinaryCrossEntropy(),new AdaGrad(0.1,0.1),5,1000, Trainer.DEFAULT_SEED,5,0.1);

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{AbstractSGDTrainer.class, AbstractFMTrainer.class,FMMultiLabelTrainer.class};
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

    private static void testTrainer(Dataset<MultiLabel> train, Dataset<MultiLabel> test, Trainer<MultiLabel> trainer) {
        Model<MultiLabel> model = trainer.train(train);

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
        Helpers.testModelProtoSerialization(model, MultiLabel.class, test);
    }

    @Test
    public void testOnnxSerialization() throws IOException, OrtException {
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();
        Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();
        FMMultiLabelModel model = sigmoid.train(train);

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-fm-test",".onnx");
        model.saveONNXModel("org.tribuo.multilabel.sgd.fm.test",1,onnxFile);

        OnnxTestUtils.onnxMultiLabelComparison(model,onnxFile,test,1e-5);

        onnxFile.toFile().delete();
    }
}
