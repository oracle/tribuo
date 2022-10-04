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

package org.tribuo.classification.sgd.kernel;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.math.kernel.RBF;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestSGDKernel {

    private static final KernelSVMTrainer t = new KernelSVMTrainer(new RBF(1.0),1,5,1000, Trainer.DEFAULT_SEED);

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(KernelSVMTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    public static Model<Label> testSGDKernel(Pair<Dataset<Label>,Dataset<Label>> p) {
        Model<Label> m = t.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertTrue(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertTrue(features.isEmpty());
        return m;
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        Model<Label> model = testSGDKernel(p);
        Helpers.testModelSerialization(model,Label.class);
        Helpers.testModelProtoSerialization(model, Label.class, p.getB());
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testSGDKernel(p);
    }

    @Test
    public void testSparseBinaryData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testSGDKernel(p);
    }

    @Test
    public void testInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = t.train(p.getA());
            m.predict(LabelledDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = t.train(p.getA());
            m.predict(LabelledDataGenerator.emptyExample());
        });
    }

    @Test
    public void testSetInvocationCount() {
        // Create new trainer and dataset so as not to mess with the other tests
        KernelSVMTrainer originalTrainer = new KernelSVMTrainer(new RBF(1.0),1,5,1000, Trainer.DEFAULT_SEED);
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();

        // The number of times to call train before final training.
        // Original trainer will be trained numOfInvocations + 1 times
        // New trainer will have its invocation count set to numOfInvocations then trained once
        int numOfInvocations = 2;

        // Create the first model and train it numOfInvocations + 1 times
        Model<Label> originalModel = null;
        for(int invocationCounter = 0; invocationCounter < numOfInvocations + 1; invocationCounter++){
            originalModel = originalTrainer.train(p.getA());
        }

        // Create a new model with same configuration, but set the invocation count to numOfInvocations
        // Assert that this succeeded, this means RNG will be at state where originalTrainer was before
        // it performed its last train.
        KernelSVMTrainer newTrainer = new KernelSVMTrainer(new RBF(1.0),1,5,1000, Trainer.DEFAULT_SEED);
        newTrainer.setInvocationCount(numOfInvocations);
        assertEquals(numOfInvocations,newTrainer.getInvocationCount());

        // Training newTrainer should now have the same result as if it
        // had trained numOfInvocations times previously even though it hasn't
        Model<Label> newModel = newTrainer.train(p.getA());
        assertEquals(originalTrainer.getInvocationCount(),newTrainer.getInvocationCount());
    }

    @Test
    public void testNegativeInvocationCount(){
        assertThrows(IllegalArgumentException.class, () -> {
            KernelSVMTrainer t = new KernelSVMTrainer(new RBF(1.0),1,5,1000, Trainer.DEFAULT_SEED);
            t.setInvocationCount(-1);
        });
    }

}
