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

package org.tribuo.classification.dtree;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.dtree.impurity.GiniIndex;
import org.tribuo.classification.ensemble.AdaBoostTrainer;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.common.tree.RandomForestTrainer;
import org.tribuo.dataset.DatasetView;
import org.tribuo.ensemble.BaggingTrainer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.tribuo.common.tree.AbstractCARTTrainer.MIN_EXAMPLES;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * These tests live here rather than in Classification/Core because otherwise it would induce circularity in the build.
 */
public class TestClassificationEnsembles {

    private static final CARTClassificationTrainer t = new CARTClassificationTrainer();
    private static final CARTClassificationTrainer subsamplingTree = new CARTClassificationTrainer(Integer.MAX_VALUE, MIN_EXAMPLES, 0.5f, new GiniIndex(), Trainer.DEFAULT_SEED);
    private static final AdaBoostTrainer adaT = new AdaBoostTrainer(t,10);
    private static final BaggingTrainer<Label> bagT = new BaggingTrainer<>(t,new VotingCombiner(),10);
    private static final RandomForestTrainer<Label> rfT = new RandomForestTrainer<>(subsamplingTree,new VotingCombiner(),10);

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(AdaBoostTrainer.class.getName());
        logger.setLevel(Level.WARNING);
        logger = Logger.getLogger(BaggingTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void testSingleClassTraining() {
        testSingleClassTraining(adaT);
        testSingleClassTraining(bagT);
        testSingleClassTraining(rfT);
    }

    public void testSingleClassTraining(Trainer<Label> trainer) {
        Pair<Dataset<Label>,Dataset<Label>> data = LabelledDataGenerator.denseTrainTest();

        DatasetView<Label> trainingData = DatasetView.createView(data.getA(),(Example<Label> e) -> e.getOutput().getLabel().equals("Foo"), "Foo selector");
        Model<Label> model = trainer.train(trainingData);
        LabelEvaluation evaluation = (LabelEvaluation) trainingData.getOutputFactory().getEvaluator().evaluate(model,data.getB());
        assertEquals(0.0,evaluation.accuracy(new Label("Bar")));
        assertEquals(0.0,evaluation.accuracy(new Label("Baz")));
        assertEquals(0.0,evaluation.accuracy(new Label("Quux")));
        assertEquals(1.0,evaluation.recall(new Label("Foo")));
    }

    public void testAdaBoost(Pair<Dataset<Label>,Dataset<Label>> p) {
        Model<Label> m = adaT.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
    }

    public void testBagging(Pair<Dataset<Label>,Dataset<Label>> p) {
        Model<Label> m = bagT.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
    }

    public void testRandomForest(Pair<Dataset<Label>,Dataset<Label>> p) {
        Model<Label> m = rfT.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        testAdaBoost(p);
        testBagging(p);
        testRandomForest(p);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testAdaBoost(p);
        testBagging(p);
        testRandomForest(p);
    }

    @Test
    public void testSparseBinaryData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testAdaBoost(p);
        testBagging(p);
        testRandomForest(p);
    }

}
