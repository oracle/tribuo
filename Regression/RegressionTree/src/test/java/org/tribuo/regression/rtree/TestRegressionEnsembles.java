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

package org.tribuo.regression.rtree;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.common.tree.AbstractCARTTrainer;
import org.tribuo.common.tree.RandomForestTrainer;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.tribuo.common.tree.AbstractCARTTrainer.MIN_EXAMPLES;

public class TestRegressionEnsembles {

    private static final CARTRegressionTrainer t = new CARTRegressionTrainer();
    private static final CARTRegressionTrainer subsamplingTree = new CARTRegressionTrainer(Integer.MAX_VALUE, MIN_EXAMPLES, 0.5f, new MeanSquaredError(), Trainer.DEFAULT_SEED);
    private static final BaggingTrainer<Regressor> bagT = new BaggingTrainer<>(t,new AveragingCombiner(),10);
    private static final RandomForestTrainer<Regressor> rfT = new RandomForestTrainer<>(subsamplingTree,new AveragingCombiner(),10);

    private static final CARTJointRegressionTrainer mT = new CARTJointRegressionTrainer();
    private static final CARTJointRegressionTrainer mSubsamplingTree = new CARTJointRegressionTrainer(Integer.MAX_VALUE, AbstractCARTTrainer.MIN_EXAMPLES, 0.5f, new MeanSquaredError(), false, Trainer.DEFAULT_SEED);
    private static final BaggingTrainer<Regressor> mBagT = new BaggingTrainer<>(mT,new AveragingCombiner(),10);
    private static final RandomForestTrainer<Regressor> mRfT = new RandomForestTrainer<>(mSubsamplingTree,new AveragingCombiner(),10);

    private static final RegressionEvaluator evaluator = new RegressionEvaluator();

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(BaggingTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    public void testBagging(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = bagT.train(p.getA());
        evaluator.evaluate(m,p.getB());
    }

    public void testRandomForest(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = rfT.train(p.getA());
        evaluator.evaluate(m,p.getB());
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        testBagging(p);
        testRandomForest(p);
        testMultiBagging(p);
        testMultiRandomForest(p);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testBagging(p);
        testRandomForest(p);
        testMultiBagging(p);
        testMultiRandomForest(p);
    }

    public void testMultiBagging(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = mBagT.train(p.getA());
        evaluator.evaluate(m,p.getB());
    }

    public void testMultiRandomForest(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = mRfT.train(p.getA());
        evaluator.evaluate(m,p.getB());
    }

    @Test
    public void testMultiDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
        testBagging(p);
        testRandomForest(p);
        testMultiBagging(p);
        testMultiRandomForest(p);
    }

    @Test
    public void testMultiSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();
        testBagging(p);
        testRandomForest(p);
        testMultiBagging(p);
        testMultiRandomForest(p);
    }

}
