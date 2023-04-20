/*
 * Copyright (c) 2023, Oracle and/or its affiliates. All rights reserved.
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

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockTrainer;

import java.util.logging.Level;
import java.util.logging.Logger;

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

}
