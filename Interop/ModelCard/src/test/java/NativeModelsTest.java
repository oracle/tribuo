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

import java.io.IOException;
import java.nio.file.Paths;

public class NativeModelsTest {
    @Test
    public void testClassificationModelCard() throws IOException {
        var factory = new LabelFactory();
        var csvLoader = new CSVLoader<>(factory);

        var irisHeaders = new String[]{"sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"};
        var irisSource = csvLoader.loadDataSource(Paths.get("src/test/input-data/bezdekIris.data"),"species", irisHeaders);
        var splitter = new TrainTestSplitter<>(irisSource,0.7,1L);

        var trainData = new MutableDataset<>(splitter.getTrain());
        var evalData = new MutableDataset<>(splitter.getTest());
        Trainer<Label> trainer = new LogisticRegressionTrainer();
        Model<Label> irisModel = trainer.train(trainData);

        var evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(irisModel, evalData);

        // create Model Card object and write to file
        ModelCard modelCard = new ModelCard(irisModel, evaluation);
        modelCard.addMetric("overall-accuracy", evaluation.accuracy());
        modelCard.addMetric("average-precision", evaluation.macroAveragedPrecision());
        modelCard.saveToFile("src/test/output-json/classificationModelCard.json");

        // read file and create Model Card object
        ModelCard modelCardCopy = new ModelCard("src/test/output-json/classificationModelCard.json");
        Assertions.assertEquals(modelCard.toString(), modelCardCopy.toString());
    }

    @Test
    public void testMultiLabelClassificationModelCard() throws IOException {
        var factory = new MultiLabelFactory();
        var trainSource = new LibSVMDataSource<>(Paths.get(".","src/test/input-data/yeast_train.svm"),factory);
        var evalSource = new LibSVMDataSource<>(Paths.get(".","src/test/input-data/yeast_test.svm"),factory,trainSource.isZeroIndexed(),trainSource.getMaxFeatureID());
        var trainData = new MutableDataset<>(trainSource);
        var evalData = new MutableDataset<>(evalSource);

        var trainer = new org.tribuo.multilabel.sgd.linear.LinearSGDTrainer(new BinaryCrossEntropy(),new AdaGrad(0.1,0.1),5,1000,1,Trainer.DEFAULT_SEED);
        var model = trainer.train(trainData);

        var evaluator = new MultiLabelEvaluator();
        var evaluation = evaluator.evaluate(model, evalData);

        // create Model Card object and write to file
        ModelCard modelCard = new ModelCard(model, evaluation);
        modelCard.addMetric("jaccord-score", evaluation.jaccardScore());
        modelCard.addMetric("balanced-error-rate", evaluation.balancedErrorRate());
        modelCard.saveToFile("src/test/output-json/multiClassificationModelCard.json");

        // read file and create Model Card object
        ModelCard modelCardCopy = new ModelCard("src/test/output-json/multiClassificationModelCard.json");
        Assertions.assertEquals(modelCard.toString(), modelCardCopy.toString());
    }

    @Test
    public void testRegressionModelCard() throws IOException {
        var factory = new RegressionFactory();
        var csvLoader = new CSVLoader<>(';', factory);

        var wineSource = csvLoader.loadDataSource(Paths.get("src/test/input-data/winequality-red.csv"),"quality");
        var splitter = new TrainTestSplitter<>(wineSource, 0.7f, 0L);
        Dataset<Regressor> trainData = new MutableDataset<>(splitter.getTrain());
        Dataset<Regressor> evalData = new MutableDataset<>(splitter.getTest());

        var trainer = new LinearSGDTrainer(new SquaredLoss(), SGD.getLinearDecaySGD(0.01), 10, trainData.size()/4, 1, 1L);
        var wineModel = trainer.train(trainData);
        var evaluator = new RegressionEvaluator();
        var evaluation = evaluator.evaluate(wineModel, evalData);

        // create Model Card object and write to file
        ModelCard modelCard = new ModelCard(wineModel, evaluation);
        modelCard.addMetric("average-rmse", evaluation.averageRMSE());
        modelCard.addMetric("average-r2", evaluation.averageR2());
        modelCard.saveToFile("src/test/output-json/regressionModelCard.json");

        // read file and create Model Card object
        ModelCard modelCardCopy = new ModelCard("src/test/output-json/regressionModelCard.json");
        Assertions.assertEquals(modelCard.toString(), modelCardCopy.toString());
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
        ModelCard modelCard = new ModelCard(model, evaluation);
        modelCard.addMetric("adjusted-mi", evaluation.adjustedMI());
        modelCard.addMetric("normalized-mi", evaluation.normalizedMI());
        modelCard.saveToFile("src/test/output-json/KMeansModelCard.json");

        // read file and create Model Card object
        ModelCard modelCardCopy = new ModelCard("src/test/output-json/KMeansModelCard.json");
        Assertions.assertEquals(modelCard.toString(), modelCardCopy.toString());
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
        ModelCard modelCard = new ModelCard(model, evaluation);
        modelCard.addMetric("overall-precision", evaluation.getPrecision());
        modelCard.addMetric("overall-recall", evaluation.getRecall());
        modelCard.saveToFile("src/test/output-json/anomalyModelCard.json");

        // read file and create Model Card object
        ModelCard modelCardCopy = new ModelCard("src/test/output-json/anomalyModelCard.json");
        Assertions.assertEquals(modelCard.toString(), modelCardCopy.toString());
    }
}