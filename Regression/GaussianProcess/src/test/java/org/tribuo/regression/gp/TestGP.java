/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression.gp;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.kernel.RBF;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.test.Helpers;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Paths;

public class TestGP {
    private static final GaussianProcessTrainer gpTrainer = new GaussianProcessTrainer(new RBF(1), 0.5, false);
    private static final RegressionEvaluator e = new RegressionEvaluator();

    @Test
    public void testWine() throws IOException {
        var outputFactory = new RegressionFactory();
        var gpTrainer = new GaussianProcessTrainer(new RBF(1), 0.5, false);

        var csvLoader = new CSVLoader<>(';',outputFactory);
        var wineSource = csvLoader.loadDataSource(Paths.get("../../tutorials/winequality-red.csv"),"quality");
        var splitter = new TrainTestSplitter<>(wineSource, 0.7f, 0L);
        Dataset<Regressor> trainData = new MutableDataset<>(splitter.getTrain());
        Dataset<Regressor> testData = new MutableDataset<>(splitter.getTest());

        var evaluator = new RegressionEvaluator();
        System.out.printf("Training data size = %d, number of features = %d%n",trainData.size(),trainData.getFeatureMap().size());
        System.out.printf("Testing data size = %d, number of features = %d%n",testData.size(),testData.getFeatureMap().size());

        var lrStartTime = System.currentTimeMillis();
        var model = gpTrainer.train(trainData);
        var lrEndTime = System.currentTimeMillis();
        System.out.println("Training GP took " + Util.formatDuration(lrStartTime,lrEndTime));

        var evaluation = evaluator.evaluate(model, trainData);
        System.out.println(evaluation.toString());

        var testEvaluation = evaluator.evaluate(model, testData);
        System.out.println(testEvaluation.toString());
    }

    public static Model<Regressor> testGP(Pair<Dataset<Regressor>,Dataset<Regressor>> p, GaussianProcessTrainer t) {
        Model<Regressor> gpModel = t.train(p.getA());
        RegressionEvaluation evaluation = e.evaluate(gpModel,p.getB());
        return gpModel;
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        Model<Regressor> model = testGP(p, gpTrainer);
        Helpers.testModelProtoSerialization(model, Regressor.class, p.getB());
    }

}
