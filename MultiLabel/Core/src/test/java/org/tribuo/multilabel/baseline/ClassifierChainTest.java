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

package org.tribuo.multilabel.baseline;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Prediction;
import org.tribuo.classification.baseline.DummyClassifierTrainer;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.multilabel.example.MultiLabelDataGenerator;
import org.tribuo.test.Helpers;

import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 *
 */
public class ClassifierChainTest {
    private static final MultiLabelFactory factory = new MultiLabelFactory();

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(ClassifierChainTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void testInvalidChain() {
        // Chains can be invalid in several ways
        // incorrect number of labels, duplicate labels, or labels not in the training data

        // Generate data
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();

        DummyClassifierTrainer trainer = DummyClassifierTrainer.createConstantTrainer("MONKEY");
        List<String> labelOrder;

        // Too many labels
        labelOrder = Arrays.asList("MONKEY","PUZZLE","TREE","PINE");
        ClassifierChainTrainer tooMany = new ClassifierChainTrainer(trainer, labelOrder);
        assertThrows(IllegalArgumentException.class, () -> tooMany.train(train));

        // Too few labels
        labelOrder = Arrays.asList("MONKEY","PUZZLE");
        ClassifierChainTrainer tooFew = new ClassifierChainTrainer(trainer, labelOrder);
        assertThrows(IllegalArgumentException.class, () -> tooFew.train(train));

        // Duplicate valid labels
        labelOrder = Arrays.asList("MONKEY","PUZZLE","PUZZLE");
        ClassifierChainTrainer duplicate = new ClassifierChainTrainer(trainer, labelOrder);
        assertThrows(IllegalArgumentException.class, () -> duplicate.train(train));

        // Labels not in the training data
        labelOrder = Arrays.asList("MONKEY","PUZZLE","PINE");
        ClassifierChainTrainer invalidLabels = new ClassifierChainTrainer(trainer, labelOrder);
        assertThrows(IllegalArgumentException.class, () -> invalidLabels.train(train));
    }

    @Test
    public void testValidChain() {
        // Generate data
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();
        Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();

        // Test random chain
        ClassifierChainTrainer trainer = new ClassifierChainTrainer(DummyClassifierTrainer.createMostFrequentTrainer(),1234);
        ClassifierChainModel model = trainer.train(train);

        List<Prediction<MultiLabel>> predictions = model.predict(test);
        Prediction<MultiLabel> first = predictions.get(0);
        MultiLabel trueLabel = factory.generateOutput("MONKEY,PUZZLE,TREE");
        assertEquals(trueLabel, first.getOutput(), "Predicted labels not equal");
        assertEquals(3,model.getLabelOrder().size());

        Helpers.testModelSerialization(model,MultiLabel.class);

        // Test ordered chain
        List<String> labelOrder = Arrays.asList("PUZZLE","MONKEY","TREE");
        trainer = new ClassifierChainTrainer(DummyClassifierTrainer.createMostFrequentTrainer(), labelOrder);
        model = trainer.train(train);

        predictions = model.predict(test);
        first = predictions.get(0);
        assertEquals(trueLabel, first.getOutput(), "Predicted labels not equal");
        assertEquals(labelOrder.get(0),model.getLabelOrder().get(0).getLabel());
        assertEquals(labelOrder.get(1),model.getLabelOrder().get(1).getLabel());
        assertEquals(labelOrder.get(2),model.getLabelOrder().get(2).getLabel());
        assertEquals(3,model.getLabelOrder().size());

        Helpers.testModelSerialization(model,MultiLabel.class);
        Helpers.testModelProtoSerialization(model, MultiLabel.class, test);
    }

}
