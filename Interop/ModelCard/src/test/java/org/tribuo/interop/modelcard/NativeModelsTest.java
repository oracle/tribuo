/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.modelcard;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.anomaly.evaluation.AnomalyEvaluator;
import org.tribuo.anomaly.example.GaussianAnomalyDataSource;
import org.tribuo.anomaly.libsvm.LibSVMAnomalyTrainer;
import org.tribuo.anomaly.libsvm.SVMAnomalyType;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.clustering.evaluation.ClusteringEvaluator;
import org.tribuo.clustering.example.GaussianClusterDataSource;
import org.tribuo.clustering.kmeans.KMeansTrainer;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.SVMParameters;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.datasource.LibSVMDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.math.optimisers.SGD;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.multilabel.evaluation.MultiLabelEvaluator;
import org.tribuo.multilabel.sgd.objectives.BinaryCrossEntropy;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.regression.sgd.objectives.SquaredLoss;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class NativeModelsTest {
    @Test
    public void testClassificationModelCard() throws IOException {
        var factory = new LabelFactory();
        var csvLoader = new CSVLoader<>(factory);

        var dataSource = csvLoader.loadDataSource(Paths.get("src/test/resources/classificationSampleData.csv"), "response");
        var splitter = new TrainTestSplitter<>(dataSource,0.7,1L);

        var trainData = new MutableDataset<>(splitter.getTrain());
        var evalData = new MutableDataset<>(splitter.getTest());
        Trainer<Label> trainer = new LogisticRegressionTrainer();
        Model<Label> model = trainer.train(trainData);

        var evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(model, evalData);

        // create Model Card object and write to file
        File output = File.createTempFile("output", "json");
        output.deleteOnExit();

        Map<String, Double> testingMetrics = new HashMap<>();
        testingMetrics.put("overall-accuracy", evaluation.accuracy());
        testingMetrics.put("average-precision", evaluation.macroAveragedPrecision());

        ModelCard modelCard = new ModelCard(model, evaluation, testingMetrics, new UsageDetailsBuilder().build());
        modelCard.saveToFile(output.toPath());

        // read file and create Model Card object
        ModelCard modelCardCopy = ModelCard.deserializeFromJson(output.toPath());
        Assertions.assertEquals(modelCard, modelCardCopy);
    }

    @Test
    public void testMultiLabelClassificationModelCard() throws IOException {
        var factory = new MultiLabelFactory();
        var dataSource = new LibSVMDataSource<>(Paths.get(".","src/test/resources/multiClassificationSampleData.svm"),factory);
        var trainData = new MutableDataset<>(dataSource);
        var evalData = new MutableDataset<>(dataSource);

        var trainer = new org.tribuo.multilabel.sgd.linear.LinearSGDTrainer(new BinaryCrossEntropy(),new AdaGrad(0.1,0.1),5,1000,1,Trainer.DEFAULT_SEED);
        var model = trainer.train(trainData);

        var evaluator = new MultiLabelEvaluator();
        var evaluation = evaluator.evaluate(model, evalData);

        // create Model Card object and write to file
        File output = File.createTempFile("output", "json");
        output.deleteOnExit();

        Map<String, Double> testingMetrics = new HashMap<>();
        testingMetrics.put("jaccord-score", evaluation.jaccardScore());
        testingMetrics.put("balanced-error-rate", evaluation.balancedErrorRate());

        ModelCard modelCard = new ModelCard(model, evaluation, testingMetrics);
        modelCard.saveToFile(output.toPath());

        // read file and create Model Card object
        ModelCard modelCardCopy = ModelCard.deserializeFromJson(output.toPath());
        Assertions.assertEquals(modelCard, modelCardCopy);
    }

    @Test
    public void testRegressionModelCard() throws IOException {
        var factory = new RegressionFactory();
        var csvLoader = new CSVLoader<>(',', factory);

        var dataSource = csvLoader.loadDataSource(Paths.get("src/test/resources/regressionSampleData.csv"),"response");
        var splitter = new TrainTestSplitter<>(dataSource, 0.7f, 0L);
        Dataset<Regressor> trainData = new MutableDataset<>(splitter.getTrain());
        Dataset<Regressor> evalData = new MutableDataset<>(splitter.getTest());

        var trainer = new LinearSGDTrainer(new SquaredLoss(), SGD.getLinearDecaySGD(0.01), 10, trainData.size()/4, 1, 1L);
        var model = trainer.train(trainData);
        var evaluator = new RegressionEvaluator();
        var evaluation = evaluator.evaluate(model, evalData);

        // create Model Card object and write to file
        File output = File.createTempFile("output", "json");
        output.deleteOnExit();

        Map<String, Double> testingMetrics = new HashMap<>();
        testingMetrics.put("average-rmse", evaluation.averageRMSE());
        testingMetrics.put("average-r2", evaluation.averageR2());

        ModelCard modelCard = new ModelCard(model, evaluation, testingMetrics, new UsageDetailsBuilder().build());
        modelCard.saveToFile(output.toPath());

        // read file and create Model Card object
        ModelCard modelCardCopy = ModelCard.deserializeFromJson(output.toPath());
        Assertions.assertEquals(modelCard, modelCardCopy);
    }

    @Test
    public void testKMeansModelCard() throws IOException {
        var testData = new MutableDataset<>(new GaussianClusterDataSource(500, 1L));
        var evalData = new MutableDataset<>(new GaussianClusterDataSource(500, 2L));

        var trainer = new KMeansTrainer(5, 10, KMeansTrainer.Distance.EUCLIDEAN, 1, 1);
        var model = trainer.train(testData);
        var evaluator = new ClusteringEvaluator();
        var evaluation = evaluator.evaluate(model,evalData);

        // create Model Card object and write to file
        File output = File.createTempFile("output", "json");
        output.deleteOnExit();

        Map<String, Double> testingMetrics = new HashMap<>();
        testingMetrics.put("adjusted-mi", evaluation.adjustedMI());
        testingMetrics.put("normalized-mi", evaluation.normalizedMI());

        ModelCard modelCard = new ModelCard(model, evaluation, testingMetrics, new UsageDetailsBuilder().build());
        modelCard.saveToFile(output.toPath());

        // read file and create Model Card object
        ModelCard modelCardCopy = ModelCard.deserializeFromJson(output.toPath());
        Assertions.assertEquals(modelCard, modelCardCopy);
    }

    @Test
    public void testAnomalyModelCard() throws IOException {
        var trainData = new MutableDataset<>(new GaussianAnomalyDataSource(2000,0.0f, 1L));
        var evalData = new MutableDataset<>(new GaussianAnomalyDataSource(2000,0.2f,2L));
        var params = new SVMParameters<>(new SVMAnomalyType(SVMAnomalyType.SVMMode.ONE_CLASS), KernelType.RBF);
        params.setGamma(1.0);
        params.setNu(0.1);

        var trainer = new LibSVMAnomalyTrainer(params);
        var model = trainer.train(trainData);

        var evaluator = new AnomalyEvaluator();
        var evaluation = evaluator.evaluate(model,evalData);

        // create Model Card object and write to file
        File output = File.createTempFile("output", "json");
        output.deleteOnExit();

        Map<String, Double> testingMetrics = new HashMap<>();
        testingMetrics.put("overall-precision", evaluation.getPrecision());
        testingMetrics.put("overall-recall", evaluation.getRecall());

        ModelCard modelCard = new ModelCard(model, evaluation, testingMetrics, new UsageDetailsBuilder().build());
        modelCard.saveToFile(output.toPath());

        // read file and create Model Card object
        ModelCard modelCardCopy = ModelCard.deserializeFromJson(output.toPath());
        Assertions.assertEquals(modelCard, modelCardCopy);
    }
}