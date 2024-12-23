/*
 * Copyright (c) 2023, 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.sequence;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.ensemble.WeightedEnsembleModel;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.protos.core.SequenceModelProto;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockTrainer;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class IndependentSequenceModelTest {

    @BeforeAll
    public static void suppressLogging() {
        Logger logger = Logger.getLogger(IndependentSequenceTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void serializationTest() {
        MutableSequenceDataset<MockOutput> dataset = SequenceDatasetTest.makeDataset();
        MockTrainer trainer = new MockTrainer("UNK");
        IndependentSequenceTrainer<MockOutput> sequenceTrainer = new IndependentSequenceTrainer<>(trainer);

        IndependentSequenceModel<MockOutput> model = sequenceTrainer.train(dataset);

        Helpers.testSequenceModelProtoSerialization(model, MockOutput.class, dataset);
    }

    @Test
    public void test431Protobufs() throws IOException, URISyntaxException {
        MutableSequenceDataset<MockOutput> dataset = SequenceDatasetTest.makeDataset();
        MockTrainer trainer = new MockTrainer("UNK");
        IndependentSequenceTrainer<MockOutput> sequenceTrainer = new IndependentSequenceTrainer<>(trainer);
        IndependentSequenceModel<MockOutput> model = sequenceTrainer.train(dataset);

        Path path = Paths.get(IndependentSequenceModelTest.class.getResource("independent-sequence-model-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(path)) {
            SequenceModelProto proto = SequenceModelProto.parseFrom(fis);
            @SuppressWarnings("unchecked")
            IndependentSequenceModel<MockOutput> deserModel = (IndependentSequenceModel<MockOutput>) SequenceModel.deserialize(proto);

            assertEquals("4.3.1", deserModel.getProvenance().getTribuoVersion());

            // As it's a mocked model underneath we run basic machinery tests to ensure the model deserialized, but
            // don't check the output as it is gibberish.
            List<List<Prediction<MockOutput>>> output = model.predict(dataset);
            List<List<Prediction<MockOutput>>> deserOutput = deserModel.predict(dataset);
            assertEquals(output.size(), deserOutput.size());
            assertEquals(dataset.size(), deserOutput.size());
        }
    }

    public void generateProtobufs() throws IOException {
        MutableSequenceDataset<MockOutput> dataset = SequenceDatasetTest.makeDataset();
        MockTrainer trainer = new MockTrainer("UNK");
        IndependentSequenceTrainer<MockOutput> sequenceTrainer = new IndependentSequenceTrainer<>(trainer);
        IndependentSequenceModel<MockOutput> model = sequenceTrainer.train(dataset);

        Helpers.writeProtobuf(model, Paths.get("src","test","resources","org","tribuo","sequence","independent-sequence-model-431.tribuo"));
    }
}
