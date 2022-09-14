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

package org.tribuo.classification.dtree;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.dtree.impurity.GiniIndex;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.common.tree.TreeModel;
import org.tribuo.dataset.DatasetView;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestCART {

    private static final CARTClassificationTrainer t = new CARTClassificationTrainer();
    private static final CARTClassificationTrainer randomt = new CARTClassificationTrainer(5,2, 0.0f,1.0f, true,
            new GiniIndex(), Trainer.DEFAULT_SEED);

    public static Model<Label> testCART(Pair<Dataset<Label>,Dataset<Label>> p, CARTClassificationTrainer trainer) {
        TreeModel<Label> m = trainer.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        Helpers.testModelProtoSerialization(m, Label.class, p.getB());

        return m;
    }

    public static void runSingleClassTraining(CARTClassificationTrainer trainer) {
        Pair<Dataset<Label>,Dataset<Label>> data = LabelledDataGenerator.denseTrainTest();

        DatasetView<Label> trainingData = DatasetView.createView(data.getA(),(Example<Label> e) -> e.getOutput().getLabel().equals("Foo"), "Foo selector");
        Model<Label> model = trainer.train(trainingData);
        LabelEvaluation evaluation = (LabelEvaluation) trainingData.getOutputFactory().getEvaluator().evaluate(model,data.getB());
        assertEquals(0.0,evaluation.accuracy(new Label("Bar")));
        assertEquals(0.0,evaluation.accuracy(new Label("Baz")));
        assertEquals(0.0,evaluation.accuracy(new Label("Quux")));
        assertEquals(1.0,evaluation.recall(new Label("Foo")));
    }

    @Test
    public void testSingleClassTraining() {
        runSingleClassTraining(t);
    }

    @Test
    public void testRandomSingleClassTraining() {
        runSingleClassTraining(randomt);
    }

    public Model<Label> runDenseData(CARTClassificationTrainer trainer) {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        return testCART(p, trainer);
    }

    @Test
    public void testDenseData() {
        Model<Label> model = runDenseData(t);
        Helpers.testModelSerialization(model,Label.class);
    }

    @Test
    public void testRandomDenseData() {
        runDenseData(randomt);
    }

    public void runSparseData(CARTClassificationTrainer trainer) {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testCART(p, trainer);
    }

    @Test
    public void testSparseData() {
        runSparseData(t);
    }

    @Test
    public void testRandomSparseData() {
        runSparseData(randomt);
    }

    public void runSparseBinaryData(CARTClassificationTrainer trainer) {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testCART(p, trainer);
    }

    @Test
    public void testSparseBinaryData() {
        runSparseBinaryData(t);
    }

    @Test
    public void testRandomSparseBinaryData() {
        runSparseBinaryData(randomt);
    }

    public static void runInvalidExample(CARTClassificationTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = trainer.train(p.getA());
            m.predict(LabelledDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testInvalidExample() {
        runInvalidExample(t);
    }

    @Test
    public void testRandomInvalidExample() {
        runInvalidExample(randomt);
    }

    public static void runEmptyExample(CARTClassificationTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = trainer.train(p.getA());
            m.predict(LabelledDataGenerator.emptyExample());
        });
    }

    @Test
    public void testEmptyExample() {
        runEmptyExample(t);
    }

    @Test
    public void testRandomEmptyExample() {
        runEmptyExample(randomt);
    }

    @Test
    public void testSetInvocationCount() {
        // Create new trainer and dataset so as not to mess with the other tests
        CARTClassificationTrainer originalTrainer = new CARTClassificationTrainer(5,2, 0.0f,1.0f, true,
                new GiniIndex(), Trainer.DEFAULT_SEED);
        Pair<Dataset<Label>,Dataset<Label>> data = LabelledDataGenerator.denseTrainTest();
        DatasetView<Label> trainingData = DatasetView.createView(data.getA(),(Example<Label> e) -> e.getOutput().getLabel().equals("Foo"), "Foo selector");

        // The number of times to call train before final training.
        // Original trainer will be trained numOfInvocations + 1 times
        // New trainer will have its invocation count set to numOfInvocations then trained once
        int numOfInvocations = 2;

        // Create the first model and train it numOfInvocations + 1 times
        TreeModel<Label> originalModel = null;
        for(int invocationCounter = 0; invocationCounter < numOfInvocations + 1; invocationCounter++){
            originalModel = originalTrainer.train(trainingData);
        }

        // Create a new model with same configuration, but set the invocation count to numOfInvocations
        // Assert that this succeeded, this means RNG will be at state where originalTrainer was before
        // it performed its last train.
        CARTClassificationTrainer newTrainer = new CARTClassificationTrainer(5,2, 0.0f,1.0f, true,
                new GiniIndex(), Trainer.DEFAULT_SEED);
        newTrainer.setInvocationCount(numOfInvocations);
        assertEquals(numOfInvocations,newTrainer.getInvocationCount());

        // Training newTrainer should now have the same result as if it
        // had trained numOfInvocations times previously even though it hasn't
        TreeModel<Label> newModel = newTrainer.train(trainingData);
        assertEquals(originalTrainer.getInvocationCount(),newTrainer.getInvocationCount());
    }

    @Test
    public void testNegativeInvocationCount(){
        assertThrows(IllegalArgumentException.class, () -> {
            CARTClassificationTrainer t = new CARTClassificationTrainer(5,2, 0.0f,1.0f, true,
                    new GiniIndex(), Trainer.DEFAULT_SEED);
            t.setInvocationCount(-1);
        });
    }
}
