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

package org.tribuo.classification.sgd.crf;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.sequence.example.SequenceDataGenerator;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.protos.core.SequenceModelProto;
import org.tribuo.sequence.SequenceDataset;
import org.tribuo.sequence.SequenceModel;
import org.tribuo.test.Helpers;

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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CRFTrainerTest {

    private static final CRFTrainer t = new CRFTrainer(new AdaGrad(0.1, 0.1), 5, 1000, Trainer.DEFAULT_SEED);

    private static final LabelFactory labelFactory = new LabelFactory();

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(CRFTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void testValidExample() {
        SequenceDataset<Label> p = SequenceDataGenerator.generateGorillaDataset(5);
        SequenceModel<Label> m = t.train(p);
        m.predict(p.getExample(0));
        Helpers.testSequenceModelProtoSerialization(m,Label.class,p);
    }

    @Test
    public void testInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            SequenceDataset<Label> p = SequenceDataGenerator.generateGorillaDataset(5);
            SequenceModel<Label> m = t.train(p);
            m.predict(SequenceDataGenerator.generateInvalidExample());
        });
    }

    @Test
    public void testOtherInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            SequenceDataset<Label> p = SequenceDataGenerator.generateGorillaDataset(5);
            SequenceModel<Label> m = t.train(p);
            m.predict(SequenceDataGenerator.generateOtherInvalidExample());
        });
    }

    @Test
    public void testEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            SequenceDataset<Label> p = SequenceDataGenerator.generateGorillaDataset(5);
            SequenceModel<Label> m = t.train(p);
            m.predict(SequenceDataGenerator.generateEmptyExample());
        });
    }

    @Test
    public void loadProtobufModel() throws IOException, URISyntaxException {
        Path path = Paths.get(CRFTrainerTest.class.getResource("crf-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(path)) {
            SequenceModelProto proto = SequenceModelProto.parseFrom(fis);
            @SuppressWarnings("unchecked")
            SequenceModel<Label> deserModel = (SequenceModel<Label>) SequenceModel.deserialize(proto);

            assertEquals("4.3.1", deserModel.getProvenance().getTribuoVersion());

            SequenceDataset<Label> p = SequenceDataGenerator.generateGorillaDataset(5);

            List<List<Prediction<Label>>> deserOutput = deserModel.predict(p);

            CRFTrainer trainer = new CRFTrainer(new AdaGrad(0.1, 0.1), 5, 1000, Trainer.DEFAULT_SEED);
            SequenceModel<Label> model = trainer.train(p);
            List<List<Prediction<Label>>> output = model.predict(p);

            assertEquals(deserOutput.size(), p.size());
            for (int i = 0; i < deserOutput.size(); i++) {
                List<Prediction<Label>> curDeser = deserOutput.get(i);
                List<Prediction<Label>> curOutput = output.get(i);
                assertTrue(Helpers.predictionListDistributionEquals(curDeser, curOutput, 1e-7));
            }
        }
    }

    /**
     * Test protobuf generation method.
     * @throws IOException If the write failed.
     */
    public void generateModel() throws IOException {
        SequenceDataset<Label> p = SequenceDataGenerator.generateGorillaDataset(5);
        CRFTrainer trainer = new CRFTrainer(new AdaGrad(0.1, 0.1), 5, 1000, Trainer.DEFAULT_SEED);
        SequenceModel<Label> model = trainer.train(p);
        Helpers.writeProtobuf(model, Paths.get("src","test","resources","org","tribuo","classification","sgd","crf","crf-431.tribuo"));
    }
}
