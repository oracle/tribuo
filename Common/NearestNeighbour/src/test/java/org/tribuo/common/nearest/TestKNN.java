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

package org.tribuo.common.nearest;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.example.DemoLabelDataSource;
import org.tribuo.classification.example.NoisyInterlockingCrescentsDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.neighbour.NeighboursQueryFactory;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForceFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Unit tests with generated datasets for KNN
 */
public class TestKNN {

    static final private KNNTrainer<Regressor> regressionTrainer = new KNNTrainer<>(3, DistanceType.L2.getDistance(), 2,
        new AveragingCombiner(), KNNModel.Backend.STREAMS, NeighboursQueryFactoryType.BRUTE_FORCE);
    static final private KNNTrainer<Label> classificationTrainer = new KNNTrainer<>(5, DistanceType.L2.getDistance(), 2,
        new VotingCombiner(), KNNModel.Backend.THREADPOOL, NeighboursQueryFactoryType.KD_TREE);

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(KNNTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void invocationCounterTest() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> pair = RegressionDataGenerator.sparseTrainTest();
        KNNTrainer<Regressor> trainer = new KNNTrainer<>(2, DistanceType.L1.getDistance(), 2, new AveragingCombiner(),
            KNNModel.Backend.THREADPOOL, NeighboursQueryFactoryType.BRUTE_FORCE);

        for (int i = 0; i < 5; i++) {
            Model<Regressor> model = trainer.train(pair.getA());
        }

        assertEquals(5,trainer.getInvocationCount());

        trainer.setInvocationCount(0);

        assertEquals(0,trainer.getInvocationCount());

        Model<Regressor> model = trainer.train(pair.getA(), Collections.emptyMap(), 3);

        assertEquals(4, trainer.getInvocationCount());
    }

    private static void testKNNRegression(KNNTrainer<Regressor> trainer) {
        Pair<Dataset<Regressor>,Dataset<Regressor>> pair = RegressionDataGenerator.denseTrainTest();

        Model<Regressor> model = trainer.train(pair.getA());

        List<Prediction<Regressor>> predictions = model.predict(pair.getB());

        assertEquals(5.0, predictions.get(0).getOutput().getValues()[0]);
        assertEquals(10.0, predictions.get(1).getOutput().getValues()[0]);
        assertEquals(20.0, predictions.get(2).getOutput().getValues()[0]);
        assertEquals(50.0, predictions.get(3).getOutput().getValues()[0]);
    }

    @Test
    public void knnRegressionMultithreadedTest() {
        testKNNRegression(regressionTrainer);
    }

    @Test
    public void knnRegressionSingleThreadedTest() {
        NeighboursQueryFactory neighboursQueryFactory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 1);
        KNNTrainer<Regressor> regressionTrainer = new KNNTrainer<>(3, 1, new AveragingCombiner(),
            KNNModel.Backend.THREADPOOL, neighboursQueryFactory);
        testKNNRegression(regressionTrainer);
    }

    private static void testKNNClassification(KNNTrainer<Label> trainer) {
        NoisyInterlockingCrescentsDataSource source = new NoisyInterlockingCrescentsDataSource(200, 1, 0.1);
        TrainTestSplitter<Label> splitter = new TrainTestSplitter<>(source, 0.8, 1L);
        MutableDataset<Label> trainingDataset = new MutableDataset<>(splitter.getTrain());
        MutableDataset<Label> testingDataset = new MutableDataset<>(splitter.getTest());

        Model<Label> model = trainer.train(trainingDataset);

        // The expected list of predictions
        List<String> expectedList = Arrays.asList("O", "X", "O", "X", "O", "X", "O", "X", "O", "X", "O", "X", "X", "O", "X", "O", "X", "O", "X", "O", "O", "X", "O", "X", "X", "X", "O", "X", "O", "O", "O", "O", "X", "O", "O", "X", "O", "X", "X", "O");

        List<Prediction<Label>> predictions = model.predict(testingDataset);
        List<String> predictionList = new ArrayList<>();
        for (Prediction<Label> prediction : predictions) {
            predictionList.add(prediction.getOutput().getLabel());
        }

        assertEquals(predictionList, expectedList);
    }

    @Test
    public void knnClassificationMultithreadedTest() {
        testKNNClassification(classificationTrainer);
    }

    @Test
    public void knnClassificationSingleThreadedTest() {
        KNNTrainer<Label> classificationTrainer = new KNNTrainer<>(5, DistanceType.L2.getDistance(), 1,
            new VotingCombiner(), KNNModel.Backend.INNERTHREADPOOL, NeighboursQueryFactoryType.KD_TREE);
        testKNNClassification(classificationTrainer);
    }

    @Test
    public void knnClassificationEvaluationTest() {
        NoisyInterlockingCrescentsDataSource source = new NoisyInterlockingCrescentsDataSource(400, 1, 0.1);
        TrainTestSplitter<Label> splitter = new TrainTestSplitter<>(source, 0.8, 1L);
        MutableDataset<Label> trainingDataset = new MutableDataset<>(splitter.getTrain());
        MutableDataset<Label> testingDataset = new MutableDataset<>(splitter.getTest());

        Model<Label> model = classificationTrainer.train(trainingDataset);

        LabelEvaluation evaluation = (LabelEvaluation) trainingDataset.getOutputFactory().getEvaluator().evaluate(model,testingDataset);

        assertEquals(evaluation.accuracy(DemoLabelDataSource.FIRST_CLASS), 1.0);
        assertEquals(evaluation.accuracy(DemoLabelDataSource.SECOND_CLASS), 1.0);
        assertEquals(evaluation.recall(DemoLabelDataSource.FIRST_CLASS), 1.0);
        assertEquals(evaluation.recall(DemoLabelDataSource.SECOND_CLASS), 1.0);

        // Test serialization
        Helpers.testModelSerialization(model, Label.class);
        Helpers.testModelProtoSerialization(model, Label.class, testingDataset);
    }

    @Test
    public void knnRegressionSparseDataTest() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> pair = RegressionDataGenerator.sparseTrainTest();
        Model<Regressor> model = regressionTrainer.train(pair.getA());
        RegressionEvaluator evaluator = new RegressionEvaluator();
        evaluator.evaluate(model, pair.getB());

        // Test serialization
        Helpers.testModelSerialization(model,Regressor.class);
        Helpers.testModelProtoSerialization(model, Regressor.class, pair.getB());
    }

    @Test
    public void knnRegressionEmptyExampleTest() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> pair = RegressionDataGenerator.denseTrainTest();
            Model<Regressor> model = regressionTrainer.train(pair.getA());
            model.predict(RegressionDataGenerator.emptyExample());
        });
    }

    @Test
    public void knnRegressionInvalidExampleTest() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> pair = RegressionDataGenerator.sparseTrainTest();
            Model<Regressor> model = regressionTrainer.train(pair.getA());
            model.predict(RegressionDataGenerator.invalidSparseExample());
        });
    }

    @Test
    @SuppressWarnings("unchecked")
    public void deserializeKNNRegressionV42ModelTest() {
        String serializedModelFilename = "KNNTrainerRegressor_k3_L2_nt2_voting_streams_v4.2.model";
        URL serializedModelPath = this.getClass().getClassLoader().getResource(serializedModelFilename);

        KNNModel<Regressor> model = null;
        try (ObjectInputStream oin = new ObjectInputStream(serializedModelPath.openStream())) {
            Object data = oin.readObject();
            model = (KNNModel<Regressor>) data;
            if (!model.validate(Regressor.class)) {
                fail("This is not a Regression model.");
            }
        } catch (IOException e) {
            fail("There is a problem accessing the serialized model file " + serializedModelPath);
        } catch (ClassNotFoundException e) {
            fail("There is a problem deserializing the model file "  + serializedModelPath);
        }

        Pair<Dataset<Regressor>,Dataset<Regressor>> pair = RegressionDataGenerator.denseTrainTest();

        List<Prediction<Regressor>> predictions = model.predict(pair.getB());

        assertEquals(5.0, predictions.get(0).getOutput().getValues()[0]);
        assertEquals(10.0, predictions.get(1).getOutput().getValues()[0]);
        assertEquals(20.0, predictions.get(2).getOutput().getValues()[0]);
        assertEquals(50.0, predictions.get(3).getOutput().getValues()[0]);
    }

}
