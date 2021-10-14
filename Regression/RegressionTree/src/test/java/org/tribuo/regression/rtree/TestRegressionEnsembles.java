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
import org.tribuo.common.tree.ExtraTreesTrainer;
import org.tribuo.common.tree.RandomForestTrainer;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.tribuo.common.tree.AbstractCARTTrainer.MIN_EXAMPLES;

public class TestRegressionEnsembles {

    private static final CARTRegressionTrainer t = new CARTRegressionTrainer();
    private static final CARTRegressionTrainer subsamplingTree = new CARTRegressionTrainer(Integer.MAX_VALUE,
            MIN_EXAMPLES, 0.0f, 0.5f, false, new MeanSquaredError(), Trainer.DEFAULT_SEED);
    private static final CARTRegressionTrainer randomTree = new CARTRegressionTrainer(Integer.MAX_VALUE, MIN_EXAMPLES
            , 0.0f,0.5f, true, new MeanSquaredError(), Trainer.DEFAULT_SEED);
    private static final BaggingTrainer<Regressor> bagT = new BaggingTrainer<>(t,new AveragingCombiner(),10);
    private static final RandomForestTrainer<Regressor> rfT = new RandomForestTrainer<>(subsamplingTree,new AveragingCombiner(),10);
    private static final ExtraTreesTrainer<Regressor> eTT = new ExtraTreesTrainer<>(randomTree,new AveragingCombiner(),10);

    private static final CARTJointRegressionTrainer mT = new CARTJointRegressionTrainer();
    private static final CARTJointRegressionTrainer mSubsamplingTree = new CARTJointRegressionTrainer(Integer.MAX_VALUE, AbstractCARTTrainer.MIN_EXAMPLES, 0.0f,0.5f, false, new MeanSquaredError(), false, Trainer.DEFAULT_SEED);
    private static final CARTJointRegressionTrainer mRandomTree = new CARTJointRegressionTrainer(Integer.MAX_VALUE,
            AbstractCARTTrainer.MIN_EXAMPLES, 0.0f,0.5f, true, new MeanSquaredError(), false, Trainer.DEFAULT_SEED);
    private static final BaggingTrainer<Regressor> mBagT = new BaggingTrainer<>(mT,new AveragingCombiner(),10);
    private static final RandomForestTrainer<Regressor> mRfT = new RandomForestTrainer<>(mSubsamplingTree,new AveragingCombiner(),10);
    private static final ExtraTreesTrainer<Regressor> mETT = new ExtraTreesTrainer<>(mRandomTree,
            new AveragingCombiner(),10);

    private static final RegressionEvaluator evaluator = new RegressionEvaluator();

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(BaggingTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    public static Model<Regressor> testBagging(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = bagT.train(p.getA());
        evaluator.evaluate(m,p.getB());
        return m;
    }

    public static Model<Regressor> testRandomForest(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = rfT.train(p.getA());
        evaluator.evaluate(m,p.getB());
        return m;
    }

    public static Model<Regressor> testExtraTrees(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = eTT.train(p.getA());
        evaluator.evaluate(m,p.getB());
        return m;
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        Model<Regressor> bagging = testBagging(p);
        Helpers.testModelSerialization(bagging,Regressor.class);
        Model<Regressor> mBagging = testMultiBagging(p);
        Helpers.testModelSerialization(mBagging,Regressor.class);
        Model<Regressor> rf = testRandomForest(p);
        Helpers.testModelSerialization(rf,Regressor.class);
        Model<Regressor> mRF = testMultiRandomForest(p);
        Helpers.testModelSerialization(mRF,Regressor.class);
        Model<Regressor> extra = testExtraTrees(p);
        Helpers.testModelSerialization(extra,Regressor.class);
        Model<Regressor> mExtra = testMultiExtraTrees(p);
        Helpers.testModelSerialization(mExtra,Regressor.class);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testBagging(p);
        testRandomForest(p);
        testMultiBagging(p);
        testMultiRandomForest(p);
        testExtraTrees(p);
        testMultiExtraTrees(p);
    }

    public static Model<Regressor> testMultiBagging(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = mBagT.train(p.getA());
        evaluator.evaluate(m,p.getB());
        return m;
    }

    public static Model<Regressor> testMultiRandomForest(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = mRfT.train(p.getA());
        evaluator.evaluate(m,p.getB());
        return m;
    }

    public static Model<Regressor> testMultiExtraTrees(Pair<Dataset<Regressor>,Dataset<Regressor>> p) {
        Model<Regressor> m = mETT.train(p.getA());
        evaluator.evaluate(m,p.getB());
        return m;
    }

    @Test
    public void testMultiDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
        testBagging(p);
        testRandomForest(p);
        testMultiBagging(p);
        testMultiRandomForest(p);
        testMultiExtraTrees(p);
    }

    @Test
    public void testMultiSparseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();
        testBagging(p);
        testRandomForest(p);
        testMultiBagging(p);
        testMultiRandomForest(p);
        testMultiExtraTrees(p);
    }

    @Test
    public void testThreeDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
        BaggingTrainer<Regressor> bagT = new BaggingTrainer<>(t,new AveragingCombiner(),10);
        Model<Regressor> llModel = bagT.train(p.getA());
        RegressionEvaluation llEval = evaluator.evaluate(llModel,p.getB());
        double expectedDim1 = 0.1632337913237244;
        double expectedDim2 = 0.1632337913237244;
        double expectedDim3 = -0.5727741047992028;
        double expectedAve = -0.08210217405058466;

        assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,llEval.averageR2(),1e-6);

        p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, true);

        // reset RNG
        bagT = new BaggingTrainer<>(t,new AveragingCombiner(),10);
        llModel = bagT.train(p.getA());
        llEval = evaluator.evaluate(llModel,p.getB());

        assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,llEval.averageR2(),1e-6);
    }
}
