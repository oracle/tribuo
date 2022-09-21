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

package org.tribuo.classification.sgd.crf;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.sequence.example.SequenceDataGenerator;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.sequence.SequenceDataset;
import org.tribuo.sequence.SequenceModel;
import org.tribuo.test.Helpers;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertThrows;

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
        Helpers.testSequenceModelSerialization(m,Label.class);
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
}
