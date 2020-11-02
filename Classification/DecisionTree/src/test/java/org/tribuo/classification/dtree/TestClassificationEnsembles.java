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

import com.oracle.labs.mlrg.olcut.config.PropertyException;
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
import org.tribuo.common.tree.ExtraTreesTrainer;
import org.tribuo.common.tree.RandomForestTrainer;
import org.tribuo.dataset.DatasetView;
import org.tribuo.ensemble.BaggingTrainer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.tribuo.common.tree.AbstractCARTTrainer.MIN_EXAMPLES;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * These tests live here rather than in Classification/Core because otherwise it would induce circularity in the build.
 */
public class TestClassificationEnsembles {

    private static final CARTClassificationTrainer t = new CARTClassificationTrainer();
    private static final CARTClassificationTrainer subsamplingTree = new CARTClassificationTrainer(Integer.MAX_VALUE,
            MIN_EXAMPLES,0.0f,  0.5f, false, new GiniIndex(), Trainer.DEFAULT_SEED);
    private static final CARTClassificationTrainer randomTree = new CARTClassificationTrainer(Integer.MAX_VALUE,
            MIN_EXAMPLES, 0.0f, 0.5f, true, new GiniIndex(), Trainer.DEFAULT_SEED);
    private static final AdaBoostTrainer adaT = new AdaBoostTrainer(t,10);
    private static final BaggingTrainer<Label> bagT = new BaggingTrainer<>(t,new VotingCombiner(),10);
    private static final RandomForestTrainer<Label> rfT = new RandomForestTrainer<>(subsamplingTree,new VotingCombiner(),10);
    private static final ExtraTreesTrainer<Label> eTT = new ExtraTreesTrainer<>(randomTree,new VotingCombiner(),
            10);

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
        testSingleClassTraining(eTT);
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

    public Model<Label> testAdaBoost(Pair<Dataset<Label>,Dataset<Label>> p) {
        Model<Label> m = adaT.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        return m;
    }

    public Model<Label> testBagging(Pair<Dataset<Label>,Dataset<Label>> p) {
        Model<Label> m = bagT.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        return m;
    }

    public Model<Label> testRandomForest(Pair<Dataset<Label>,Dataset<Label>> p) {
        Model<Label> m = rfT.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        return m;
    }

    public Model<Label> testExtraTrees(Pair<Dataset<Label>,Dataset<Label>> p) {
        Model<Label> m = eTT.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        return m;
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();

        Model<Label> boost = testAdaBoost(p);
        Helpers.testModelSerialization(boost,Label.class);

        Model<Label> bag = testBagging(p);
        Helpers.testModelSerialization(bag,Label.class);

        Model<Label> rf = testRandomForest(p);
        Helpers.testModelSerialization(rf,Label.class);

        Model<Label> extra = testExtraTrees(p);
        Helpers.testModelSerialization(extra,Label.class);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testAdaBoost(p);
        testBagging(p);
        testRandomForest(p);
        testExtraTrees(p);
    }

    @Test
    public void testSparseBinaryData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testAdaBoost(p);
        testBagging(p);
        testRandomForest(p);
        testExtraTrees(p);
    }

    @Test
    public void testInvalidExtraTrees() {
        assertThrows(PropertyException.class, () -> {
            new ExtraTreesTrainer<>(subsamplingTree,new VotingCombiner(),
                    10);
        });
    }

    @Test
    public void testInvalidRandomForest() {
        assertThrows(PropertyException.class, () -> {
            new RandomForestTrainer<>(randomTree,new VotingCombiner(),
                    10);
        });
    }
}
