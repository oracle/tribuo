/*
 * Copyright (c) 2015, 2023, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.baseline;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.evaluation.Evaluator;
import org.junit.jupiter.api.Test;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestDummyClassifier {

    private static final Trainer<Label> constant = DummyClassifierTrainer.createConstantTrainer("Foo");
    private static final Trainer<Label> uniform = DummyClassifierTrainer.createUniformTrainer(1L);
    private static final Trainer<Label> stratified = DummyClassifierTrainer.createStratifiedTrainer(1L);
    private static final Trainer<Label> mostFrequent = DummyClassifierTrainer.createMostFrequentTrainer();

    private static final List<Trainer<Label>> trainers = Arrays.asList(constant,uniform,stratified,mostFrequent);

    public static void testDummyClassifier(Pair<Dataset<Label>, Dataset<Label>> p, boolean testModelSave) {
        for (Trainer<Label> t : trainers) {
            Model<Label> m = t.train(p.getA());
            Evaluator<Label, LabelEvaluation> evaluator = new LabelEvaluator();
            LabelEvaluation evaluation = evaluator.evaluate(m,p.getB());
            if (testModelSave) {
                Helpers.testModelProtoSerialization(m, Label.class, p.getB());
            }
        }
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        testDummyClassifier(p,true);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testDummyClassifier(p,false);
    }

    @Test
    public void testSparseBinaryData() {
        Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testDummyClassifier(p,false);
    }

    @Test
    public void loadProtobufModel() throws IOException, URISyntaxException {
        Path path = Paths.get(TestDummyClassifier.class.getResource("dummyclf-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(path)) {
            ModelProto proto = ModelProto.parseFrom(fis);
            DummyClassifierModel deserModel = (DummyClassifierModel) Model.deserialize(proto);

            assertEquals("4.3.1", deserModel.getProvenance().getTribuoVersion());

            Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            DummyClassifierTrainer trainer = DummyClassifierTrainer.createMostFrequentTrainer();
            Model<Label> model = trainer.train(p.getA());

            List<Prediction<Label>> deserPredictions = deserModel.predict(p.getB());
            List<Prediction<Label>> predictions = model.predict(p.getB());
            assertEquals(p.getB().size(), deserPredictions.size());
            assertTrue(Helpers.predictionListDistributionEquals(predictions, deserPredictions));
        }
    }

    public void generateProtobuf() throws IOException {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();

        DummyClassifierTrainer trainer = DummyClassifierTrainer.createMostFrequentTrainer();
        Model<Label> model = trainer.train(p.getA());

        Helpers.writeProtobuf(model, Paths.get("src","test","resources","org","tribuo","classification","baseline","dummyclf-431.tribuo"));
    }
}