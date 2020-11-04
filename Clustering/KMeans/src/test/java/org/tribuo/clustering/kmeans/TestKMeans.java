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

package org.tribuo.clustering.kmeans;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.evaluation.ClusteringEvaluation;
import org.tribuo.clustering.evaluation.ClusteringEvaluator;
import org.tribuo.clustering.example.ClusteringDataGenerator;
import org.tribuo.clustering.kmeans.KMeansTrainer.Distance;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Smoke tests for k-means.
 */
public class TestKMeans {

    private static final KMeansTrainer t = new KMeansTrainer(4,10,Distance.EUCLIDEAN,1,1);

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(KMeansTrainer.class.getName());
        logger.setLevel(Level.WARNING);
        logger = Logger.getLogger(org.tribuo.util.infotheory.InformationTheory.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void testEvaluation() {
        Dataset<ClusterID> data = ClusteringDataGenerator.gaussianClusters(500, 1L);
        Dataset<ClusterID> test = ClusteringDataGenerator.gaussianClusters(500, 2L);
        ClusteringEvaluator eval = new ClusteringEvaluator();

        KMeansTrainer trainer = new KMeansTrainer(5,10,Distance.EUCLIDEAN,1,1);

        KMeansModel model = trainer.train(data);

        ClusteringEvaluation trainEvaluation = eval.evaluate(model,data);
        assertFalse(Double.isNaN(trainEvaluation.adjustedMI()));
        assertFalse(Double.isNaN(trainEvaluation.normalizedMI()));

        ClusteringEvaluation testEvaluation = eval.evaluate(model,test);
        assertFalse(Double.isNaN(testEvaluation.adjustedMI()));
        assertFalse(Double.isNaN(testEvaluation.normalizedMI()));
    }

    public static Model<ClusterID> testKMeans(Pair<Dataset<ClusterID>, Dataset<ClusterID>> p) {
        Model<ClusterID> m = t.train(p.getA());
        ClusteringEvaluator e = new ClusteringEvaluator();
        e.evaluate(m,p.getB());
        return m;
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<ClusterID>,Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();
        Model<ClusterID> model = testKMeans(p);
        Helpers.testModelSerialization(model,ClusterID.class);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<ClusterID>,Dataset<ClusterID>> p = ClusteringDataGenerator.sparseTrainTest();
        testKMeans(p);
    }

    @Test
    public void testInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<ClusterID>, Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();
            Model<ClusterID> m = t.train(p.getA());
            m.predict(ClusteringDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<ClusterID>, Dataset<ClusterID>> p = ClusteringDataGenerator.denseTrainTest();
            Model<ClusterID> m = t.train(p.getA());
            m.predict(ClusteringDataGenerator.emptyExample());
        });
    }

}
