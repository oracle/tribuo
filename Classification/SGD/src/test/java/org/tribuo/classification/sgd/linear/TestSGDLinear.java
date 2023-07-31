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

package org.tribuo.classification.sgd.linear;

import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.example.InterlockingCrescentsDataSource;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.common.sgd.AbstractLinearSGDTrainer;
import org.tribuo.common.sgd.AbstractSGDTrainer;
import org.tribuo.dataset.DatasetView;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.interop.onnx.OnnxTestUtils;
import org.tribuo.math.optimisers.AdaGrad;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
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
import static org.junit.jupiter.api.Assertions.assertTrue;

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
        LinearSGDModel m = t.train(p.getA());
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
        LinearSGDModel model = t.train(p.getA());

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-sgd-test",".onnx");
        model.saveONNXModel("org.tribuo.classification.sgd.linear.test",1,onnxFile);

        OnnxTestUtils.onnxLabelComparison(model,onnxFile,p.getB(),1e-6);

        onnxFile.toFile().delete();
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        Model<Label> model = testSGDLinear(p);
        Helpers.testModelProtoSerialization(model, Label.class, p.getB());
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

    @Test
    public void testBatching() {
        LabelEvaluator eval = new LabelEvaluator();
        InterlockingCrescentsDataSource source = new InterlockingCrescentsDataSource(100);
        TrainTestSplitter<Label> split = new TrainTestSplitter<>(source);
        Dataset<Label> train = new MutableDataset<>(split.getTrain());
        Dataset<Label> test = new MutableDataset<>(split.getTest());

        LinearSGDTrainer batch = new LinearSGDTrainer(new LogMulticlass(),new AdaGrad(0.1,0.1),5,1000, 10, Trainer.DEFAULT_SEED);
        Model<Label> batchModel = batch.train(train);
        LabelEvaluation batchEval = eval.evaluate(batchModel, test);

        LinearSGDTrainer single = new LinearSGDTrainer(new LogMulticlass(),new AdaGrad(0.1,0.1),2,1000, 1, Trainer.DEFAULT_SEED);
        Model<Label> singleModel = single.train(train);
        LabelEvaluation singleEval = eval.evaluate(singleModel, test);

        assertEquals(batchEval.accuracy(), singleEval.accuracy());
    }

    @Test
    public void loadProtobufModel() throws IOException, URISyntaxException {
        Path path = Paths.get(TestSGDLinear.class.getResource("lin-clf-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(path)) {
            ModelProto proto = ModelProto.parseFrom(fis);
            @SuppressWarnings("unchecked")
            Model<Label> deserModel = (Model<Label>) Model.deserialize(proto);

            assertEquals("4.3.1", deserModel.getProvenance().getTribuoVersion());

            Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();

            List<Prediction<Label>> deserOutput = deserModel.predict(p.getB());

            LinearSGDTrainer trainer = new LinearSGDTrainer(new LogMulticlass(), new AdaGrad(1.0,0.1),10,1000, 1L);
            Model<Label> model = trainer.train(p.getA());
            List<Prediction<Label>> output = model.predict(p.getB());

            assertEquals(deserOutput.size(), p.getB().size());
            assertTrue(Helpers.predictionListDistributionEquals(deserOutput, output, 1e-7));
        }
    }

    /**
     * Test protobuf generation method.
     * @throws IOException If the write failed.
     */
    public void generateModel() throws IOException {
        Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        LinearSGDTrainer trainer = new LinearSGDTrainer(new LogMulticlass(), new AdaGrad(1.0,0.1),10,1000, 1L);
        Model<Label> model = trainer.train(p.getA());
        Helpers.writeProtobuf(model, Paths.get("src","test","resources","org","tribuo","classification","sgd","linear","lin-clf-431.tribuo"));
    }
}
