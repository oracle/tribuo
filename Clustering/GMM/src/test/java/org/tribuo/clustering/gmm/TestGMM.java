/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.clustering.gmm;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.evaluation.ClusteringEvaluation;
import org.tribuo.clustering.evaluation.ClusteringEvaluator;
import org.tribuo.clustering.example.ClusteringDataGenerator;
import org.tribuo.clustering.example.GaussianClusterDataSource;
import org.tribuo.math.distributions.MultivariateNormalDistribution;
import org.tribuo.math.la.DenseVector;
import org.tribuo.test.Helpers;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Smoke tests for GMM.
 */
public class TestGMM {

    private static final GMMTrainer diagonal = new GMMTrainer(5, 50, MultivariateNormalDistribution.CovarianceType.DIAGONAL,
            GMMTrainer.Initialisation.RANDOM, 1e-3, 1, 1);

    private static final GMMTrainer fullParallel = new GMMTrainer(5, 50, MultivariateNormalDistribution.CovarianceType.FULL,
            GMMTrainer.Initialisation.PLUSPLUS, 1e-3, 4, 1);

    private static final GMMTrainer plusPlusFull = new GMMTrainer(5, 50, MultivariateNormalDistribution.CovarianceType.FULL,
            GMMTrainer.Initialisation.PLUSPLUS, 1e-3, 1, 1);

    private static final GMMTrainer plusPlusSpherical = new GMMTrainer(5, 50, MultivariateNormalDistribution.CovarianceType.SPHERICAL,
            GMMTrainer.Initialisation.PLUSPLUS, 1e-3, 1, 1);

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(GMMTrainer.class.getName());
        logger.setLevel(Level.WARNING);
        logger = Logger.getLogger(org.tribuo.util.infotheory.InformationTheory.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void testEvaluation() {
        runEvaluation(diagonal);
    }

    @Test
    public void testPlusPlusSphericalEvaluation() {
        runEvaluation(plusPlusSpherical);
    }

    @Test
    public void testPlusPlusFullEvaluation() {
        runEvaluation(plusPlusFull);
    }

    @Test
    public void testParallelEvaluation() {
        runEvaluation(fullParallel);
    }

    public static void runEvaluation(GMMTrainer trainer) {
        Dataset<ClusterID> data = new MutableDataset<>(new GaussianClusterDataSource(500, 1L));
        Dataset<ClusterID> test = ClusteringDataGenerator.gaussianClusters(500, 2L);
        ClusteringEvaluator eval = new ClusteringEvaluator();

        GaussianMixtureModel model = trainer.train(data);

        Helpers.testModelProtoSerialization(model, ClusterID.class, test);

        ClusteringEvaluation trainEvaluation = eval.evaluate(model,data);
        assertFalse(Double.isNaN(trainEvaluation.adjustedMI()));
        assertFalse(Double.isNaN(trainEvaluation.normalizedMI()));

        ClusteringEvaluation testEvaluation = eval.evaluate(model,test);
        assertFalse(Double.isNaN(testEvaluation.adjustedMI()));
        assertFalse(Double.isNaN(testEvaluation.normalizedMI()));
    }

    public static Model<ClusterID> testTrainer(Pair<Dataset<ClusterID>, Dataset<ClusterID>> p, GMMTrainer trainer) {
        Model<ClusterID> m = trainer.train(p.getA());
        ClusteringEvaluator e = new ClusteringEvaluator();
        e.evaluate(m,p.getB());
        return m;
    }

    public static Model<ClusterID> runDenseData(GMMTrainer trainer) {
        Pair<Dataset<ClusterID>,Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();
        return testTrainer(p, trainer);
    }

    @Test
    public void testDenseData() {
        Model<ClusterID> model = runDenseData(diagonal);
        Helpers.testModelProtoSerialization(model, ClusterID.class);
    }

    @Test
    public void testPlusPlusDenseData() {
        runDenseData(plusPlusFull);
    }

    public void runSparseData(GMMTrainer trainer) {
        Pair<Dataset<ClusterID>,Dataset<ClusterID>> p = ClusteringDataGenerator.sparseTrainTest();
        testTrainer(p, trainer);
    }

    @Test
    public void testSparseData() {
        runSparseData(diagonal);
    }

    @Test
    public void testPlusPlusSparseData() {
        runSparseData(plusPlusFull);
    }

    public void runInvalidExample(GMMTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<ClusterID>, Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();
            Model<ClusterID> m = trainer.train(p.getA());
            m.predict(ClusteringDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testInvalidExample() {
        runInvalidExample(diagonal);
    }

    @Test
    public void testPlusPlusInvalidExample() {
        runInvalidExample(plusPlusFull);
    }

    public void runEmptyExample(GMMTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<ClusterID>, Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();
            Model<ClusterID> m = trainer.train(p.getA());
            m.predict(ClusteringDataGenerator.emptyExample());
        });
    }

    @Test
    public void testEmptyExample() {
        runEmptyExample(diagonal);
    }

    @Test
    public void testPlusPlusEmptyExample() {
        runEmptyExample(plusPlusFull);
    }

    @Test
    public void testPlusPlusTooManyCentroids() {
        assertThrows(IllegalArgumentException.class, () -> {
            Dataset<ClusterID> data = ClusteringDataGenerator.gaussianClusters(3, 1L);
            plusPlusFull.train(data);
        });
    }

    @Test
    public void testSetInvocationCount() {
        // Create new trainer and dataset so as not to mess with the other tests
        GMMTrainer originalTrainer = new GMMTrainer(4, 10, MultivariateNormalDistribution.CovarianceType.DIAGONAL,
                GMMTrainer.Initialisation.RANDOM, 1e-3, 1, 1);
        Pair<Dataset<ClusterID>,Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();

        // The number of times to call train before final training.
        // Original trainer will be trained numOfInvocations + 1 times
        // New trainer will have its invocation count set to numOfInvocations then trained once
        int numOfInvocations = 2;

        // Create the first model and train it numOfInvocations + 1 times
        GaussianMixtureModel originalModel = null;
        for(int invocationCounter = 0; invocationCounter < numOfInvocations + 1; invocationCounter++){
            originalModel = originalTrainer.train(p.getA());
        }

        // Create a new model with same configuration, but set the invocation count to numOfInvocations
        // Assert that this succeeded, this means RNG will be at state where originalTrainer was before
        // it performed its last train.
        GMMTrainer newTrainer = new GMMTrainer(4, 10, MultivariateNormalDistribution.CovarianceType.DIAGONAL,
                GMMTrainer.Initialisation.RANDOM, 1e-3, 1, 1);
        newTrainer.setInvocationCount(numOfInvocations);
        assertEquals(numOfInvocations,newTrainer.getInvocationCount());

        // Training newTrainer should now have the same result as if it
        // had trained numOfInvocations times previously even though it hasn't
        GaussianMixtureModel newModel = newTrainer.train(p.getA());
        assertEquals(originalTrainer.getInvocationCount(),newTrainer.getInvocationCount());

        DenseVector[] newWeights = newModel.getMeanVectors();
        DenseVector[] oldWeights = originalModel.getMeanVectors();

        for (int centroidIndex = 0; centroidIndex < newWeights.length; centroidIndex++){
            assertEquals(oldWeights[centroidIndex],newWeights[centroidIndex]);
        }
    }

    @Test
    public void testNegativeInvocationCount(){
        assertThrows(IllegalArgumentException.class, () -> {
            GMMTrainer t = new GMMTrainer(4, 10, MultivariateNormalDistribution.CovarianceType.DIAGONAL,
                    GMMTrainer.Initialisation.RANDOM, 1e-3, 1, 1);
            t.setInvocationCount(-1);
        });
    }
}
