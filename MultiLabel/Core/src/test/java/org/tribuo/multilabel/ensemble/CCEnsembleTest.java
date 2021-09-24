/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.multilabel.ensemble;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Prediction;
import org.tribuo.classification.baseline.DummyClassifierTrainer;
import org.tribuo.ensemble.WeightedEnsembleModel;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.multilabel.baseline.ClassifierChainTrainer;
import org.tribuo.multilabel.example.MultiLabelDataGenerator;
import org.tribuo.test.Helpers;

import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 *
 */
public class CCEnsembleTest {
    private static final MultiLabelFactory factory = new MultiLabelFactory();

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{
                CCEnsembleTrainer.class,
                ClassifierChainTrainer.class
        };
        for (Class<?> c : classes) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
        }
    }

    @Test
    public void testInvalidEnsemble() {
        assertThrows(IllegalArgumentException.class, () -> new CCEnsembleTrainer(DummyClassifierTrainer.createMostFrequentTrainer(),0,0));
    }

    @Test
    public void testValidChain() {
        // Generate data
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();
        Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();

        // Build model
        CCEnsembleTrainer trainer = new CCEnsembleTrainer(DummyClassifierTrainer.createMostFrequentTrainer(),10,1);
        WeightedEnsembleModel<MultiLabel> model = trainer.train(train);

        // Test basic properties
        List<Prediction<MultiLabel>> predictions = model.predict(test);
        Prediction<MultiLabel> first = predictions.get(0);
        MultiLabel trueLabel = factory.generateOutput("MONKEY,PUZZLE,TREE");
        assertEquals(trueLabel, first.getOutput(), "Predicted labels not equal");

        Helpers.testModelSerialization(model,MultiLabel.class);
    }

}
