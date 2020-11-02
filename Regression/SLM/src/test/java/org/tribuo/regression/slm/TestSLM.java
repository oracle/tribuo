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

package org.tribuo.regression.slm;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestSLM {

    private static final SLMTrainer SFS = new SLMTrainer(false,-1);
    private static final SLMTrainer SFSN = new SLMTrainer(false,-1);
    private static final ElasticNetCDTrainer ELASTIC_NET = new ElasticNetCDTrainer(1.0,0.5);
    private static final LARSTrainer LARS = new LARSTrainer(-1);
    private static final LARSLassoTrainer LARS_LASSO = new LARSLassoTrainer(-1);

    @BeforeAll
    public static void turnDownLogging() {
        Logger logger = Logger.getLogger(SLMTrainer.class.getName());
        logger.setLevel(Level.WARNING);
        logger = Logger.getLogger(ElasticNetCDTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    // This is a bit contrived, but it makes the trainer that failed appear in the stack trace.
    public static Model<Regressor> testTrainer(Trainer<Regressor> trainer, Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = trainer.train(p.getA());
        RegressionEvaluator e = new RegressionEvaluator();
        RegressionEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        return m;
    }

    public static Model<Regressor> testSFS(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        return testTrainer(SFS,p);
    }

    public static Model<Regressor> testSFSN(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        return testTrainer(SFSN,p);
    }

    public static Model<Regressor> testLARS(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        return testTrainer(LARS,p);
    }

    public static Model<Regressor> testLASSO(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        return testTrainer(LARS_LASSO,p);
    }

    public static Model<Regressor> testElasticNet(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        return testTrainer(ELASTIC_NET,p);
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        Model<Regressor> sfs = testSFS(p);
        Helpers.testModelSerialization(sfs,Regressor.class);
        Model<Regressor> sfsn = testSFSN(p);
        Helpers.testModelSerialization(sfsn,Regressor.class);
        Model<Regressor> lars = testLARS(p);
        Helpers.testModelSerialization(lars,Regressor.class);
        Model<Regressor> lasso = testLASSO(p);
        Helpers.testModelSerialization(lasso,Regressor.class);
        Model<Regressor> elastic = testElasticNet(p);
        Helpers.testModelSerialization(elastic,Regressor.class);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testSFS(p);
        testSFSN(p);
        testLARS(p);
        testLASSO(p);
        testElasticNet(p);
    }

    @Test
    public void testMultiDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
        testSFS(p);
        testSFSN(p);
        testLARS(p);
        testLASSO(p);
        testElasticNet(p);
    }

    @Test
    public void testMultiSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();
        testSFS(p);
        testSFSN(p);
        testLARS(p);
        testLASSO(p);
        testElasticNet(p);
    }

}
