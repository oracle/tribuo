/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.multilabel;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.sgd.linear.LinearSGDTrainer;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.multilabel.baseline.IndependentMultiLabelTrainer;
import org.tribuo.multilabel.example.MultiLabelDataGenerator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 *
 */
public class IndependentMultiLabelTest {

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(LinearSGDTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void testIndependentBinaryPredictions() {
        MultiLabelFactory factory = new MultiLabelFactory();
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();
        Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();

        IndependentMultiLabelTrainer trainer = new IndependentMultiLabelTrainer(new LogisticRegressionTrainer());
        Model<MultiLabel> model = trainer.train(train);

        List<Prediction<MultiLabel>> predictions = model.predict(test);
        Prediction<MultiLabel> first = predictions.get(0);
        MultiLabel trueLabel = factory.generateOutput("MONKEY,PUZZLE,TREE");
        assertEquals(trueLabel, first.getOutput(), "Predicted labels not equal");
        Map<String, List<Pair<String,Double>>> features = model.getTopFeatures(2);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());

        Helpers.testModelSerialization(model,MultiLabel.class);
    }

}
