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

package org.tribuo.multilabel.baseline;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.baseline.DummyClassifierTrainer;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.multilabel.example.MultiLabelDataGenerator;
import org.junit.jupiter.api.Assertions;
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

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 *
 */
public class IndependentMultiLabelTest {

    @Test
    public void testIndependentBinaryPredictions() {
        MultiLabelFactory factory = new MultiLabelFactory();
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();
        Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();

        IndependentMultiLabelTrainer trainer = new IndependentMultiLabelTrainer(DummyClassifierTrainer.createMostFrequentTrainer());
        Model<MultiLabel> model = trainer.train(train);

        List<Prediction<MultiLabel>> predictions = model.predict(test);
        Prediction<MultiLabel> first = predictions.get(0);
        MultiLabel trueLabel = factory.generateOutput("MONKEY,PUZZLE,TREE");
        assertEquals(trueLabel, first.getOutput(), "Predicted labels not equal");
        Map<String, List<Pair<String,Double>>> features = model.getTopFeatures(2);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());

        Helpers.testModelProtoSerialization(model, MultiLabel.class, test);
    }

    @Test
    public void loadProtobufModel() throws IOException, URISyntaxException {
        Path path = Paths.get(IndependentMultiLabelTest.class.getResource("iml-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(path)) {
            ModelProto proto = ModelProto.parseFrom(fis);
            IndependentMultiLabelModel model = (IndependentMultiLabelModel) Model.deserialize(proto);

            assertEquals("4.3.1", model.getProvenance().getTribuoVersion());

            MultiLabelFactory factory = new MultiLabelFactory();
            Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();
            List<Prediction<MultiLabel>> predictions = model.predict(test);
            Prediction<MultiLabel> first = predictions.get(0);
            MultiLabel trueLabel = factory.generateOutput("MONKEY,PUZZLE,TREE");
            assertEquals(trueLabel, first.getOutput(), "Predicted labels not equal");
        }
    }

    public void generateProtobuf() throws IOException {
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();

        IndependentMultiLabelTrainer trainer = new IndependentMultiLabelTrainer(DummyClassifierTrainer.createMostFrequentTrainer());
        Model<MultiLabel> model = trainer.train(train);

        Helpers.writeProtobuf(model, Paths.get("src","test","resources","org","tribuo","multilabel","baseline","iml-431.tribuo"));
    }

}
