/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.clustering.kmeans;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.evaluation.ClusteringEvaluation;
import org.tribuo.clustering.evaluation.ClusteringEvaluator;
import org.tribuo.clustering.example.ClusteringDataGenerator;
import org.tribuo.clustering.example.GaussianClusterDataSource;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.DenseVector;
import org.tribuo.test.Helpers;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Smoke tests for k-means.
 */
public class TestKMeans {

    private static final KMeansTrainer t = new KMeansTrainer(4,10, DistanceType.L2.getDistance(),
            KMeansTrainer.Initialisation.RANDOM, 1,1);

    private static final KMeansTrainer plusPlus = new KMeansTrainer(4,10,  DistanceType.L2.getDistance(),
            KMeansTrainer.Initialisation.PLUSPLUS, 1,1);

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(KMeansTrainer.class.getName());
        logger.setLevel(Level.WARNING);
        logger = Logger.getLogger(org.tribuo.util.infotheory.InformationTheory.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void testEvaluation() {
        runEvaluation(t);
    }

    @Test
    public void testPlusPlusEvaluation() {
        runEvaluation(plusPlus);
    }

    public static void runEvaluation(KMeansTrainer trainer) {
        Dataset<ClusterID> data = new MutableDataset<>(new GaussianClusterDataSource(500, 1L));
        Dataset<ClusterID> test = ClusteringDataGenerator.gaussianClusters(500, 2L);
        ClusteringEvaluator eval = new ClusteringEvaluator();

        KMeansModel model = trainer.train(data);

        Helpers.testModelSerialization(model, ClusterID.class);
        Helpers.testModelProtoSerialization(model, ClusterID.class, test);

        ClusteringEvaluation trainEvaluation = eval.evaluate(model,data);
        assertFalse(Double.isNaN(trainEvaluation.adjustedMI()));
        assertFalse(Double.isNaN(trainEvaluation.normalizedMI()));

        ClusteringEvaluation testEvaluation = eval.evaluate(model,test);
        assertFalse(Double.isNaN(testEvaluation.adjustedMI()));
        assertFalse(Double.isNaN(testEvaluation.normalizedMI()));
    }

    public static Model<ClusterID> testTrainer(Pair<Dataset<ClusterID>, Dataset<ClusterID>> p, KMeansTrainer trainer) {
        Model<ClusterID> m = trainer.train(p.getA());
        ClusteringEvaluator e = new ClusteringEvaluator();
        e.evaluate(m,p.getB());
        return m;
    }

    public static Model<ClusterID> runDenseData(KMeansTrainer trainer) {
        Pair<Dataset<ClusterID>,Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();
        return testTrainer(p, trainer);
    }

    @Test
    public void testDenseData() {
        Model<ClusterID> model = runDenseData(t);
        Helpers.testModelSerialization(model,ClusterID.class);
    }

    @Test
    public void testPlusPlusDenseData() {
        runDenseData(plusPlus);
    }

    public void runSparseData(KMeansTrainer trainer) {
        Pair<Dataset<ClusterID>,Dataset<ClusterID>> p = ClusteringDataGenerator.sparseTrainTest();
        testTrainer(p, trainer);
    }

    @Test
    public void testSparseData() {
        runSparseData(t);
    }

    @Test
    public void testPlusPlusSparseData() {
        runSparseData(plusPlus);
    }

    public void runInvalidExample(KMeansTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<ClusterID>, Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();
            Model<ClusterID> m = trainer.train(p.getA());
            m.predict(ClusteringDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testInvalidExample() {
        runInvalidExample(t);
    }

    @Test
    public void testPlusPlusInvalidExample() {
        runInvalidExample(plusPlus);
    }


    public void runEmptyExample(KMeansTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<ClusterID>, Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();
            Model<ClusterID> m = trainer.train(p.getA());
            m.predict(ClusteringDataGenerator.emptyExample());
        });
    }

    @Test
    public void testEmptyExample() {
        runEmptyExample(t);
    }

    @Test
    public void testPlusPlusEmptyExample() {
        runEmptyExample(plusPlus);
    }

    @Test
    public void testPlusPlusTooManyCentroids() {
        assertThrows(IllegalArgumentException.class, () -> {
            Dataset<ClusterID> data = ClusteringDataGenerator.gaussianClusters(3, 1L);
            plusPlus.train(data);
        });
    }

    @Test
    public void testSetInvocationCount() {
        // Create new trainer and dataset so as not to mess with the other tests
        KMeansTrainer originalTrainer = new KMeansTrainer(4,10, DistanceType.L2.getDistance(),
                KMeansTrainer.Initialisation.RANDOM, 1,1);
        Pair<Dataset<ClusterID>,Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();

        // The number of times to call train before final training.
        // Original trainer will be trained numOfInvocations + 1 times
        // New trainer will have it's invocation count set to numOfInvocations then trained once
        int numOfInvocations = 2;

        // Create the first model and train it numOfInvocations + 1 times
        KMeansModel originalModel = null;
        for(int invocationCounter = 0; invocationCounter < numOfInvocations + 1; invocationCounter++){
            originalModel = originalTrainer.train(p.getA());
        }

        // Create a new model with same configuration, but set the invocation count to numOfInvocations
        // Assert that this succeeded, this means RNG will be at state where originalTrainer was before
        // it performed its last train.
        KMeansTrainer newTrainer = new KMeansTrainer(4,10, DistanceType.L2.getDistance(),
                KMeansTrainer.Initialisation.RANDOM, 1,1);
        newTrainer.setInvocationCount(numOfInvocations);
        assertEquals(numOfInvocations,newTrainer.getInvocationCount());

        // Training newTrainer should now have the same result as if it
        // had trained numOfInvocations times previously even though it hasn't
        KMeansModel newModel = newTrainer.train(p.getA());
        assertEquals(originalTrainer.getInvocationCount(),newTrainer.getInvocationCount());

        DenseVector[] newWeights = newModel.getCentroidVectors();
        DenseVector[] oldWeights = originalModel.getCentroidVectors();

        for (int centroidIndex = 0; centroidIndex < newWeights.length; centroidIndex++){
            assertEquals(oldWeights[centroidIndex],newWeights[centroidIndex]);
        }
    }

    @Test
    public void testNegativeInvocationCount(){
        assertThrows(IllegalArgumentException.class, () -> {
            KMeansTrainer t = new KMeansTrainer(4,10, DistanceType.L2.getDistance(),
                    KMeansTrainer.Initialisation.RANDOM, 1,1);
            t.setInvocationCount(-1);
        });
    }

    @Test
    public void testToString(){
        assertEquals("KMeansTrainer(centroids=4,distance=L2Distance(),seed=1,numThreads=1, initialisationType=RANDOM)", t.toString());
        assertEquals("KMeansTrainer(centroids=4,distance=L2Distance(),seed=1,numThreads=1, initialisationType=PLUSPLUS)", plusPlus.toString());
    }
}
